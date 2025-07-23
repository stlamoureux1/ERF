#include "ERF_SatAdj.H"

using namespace amrex;

/**
 * Compute Precipitation-related Microphysics quantities.
 */
void SatAdj::AdvanceSatAdj (const SolverChoice& /*solverChoice*/)
{
    auto tabs  = mic_fab_vars[MicVar_SatAdj::tabs];

    // Expose for GPU
    Real d_fac_cond = m_fac_cond;
    Real rdOcp      = m_rdOcp;

    // get the temperature, density, theta, qt and qc from input
    for ( MFIter mfi(*tabs,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const auto& tbx = mfi.tilebox();

        auto qv_array    = mic_fab_vars[MicVar_SatAdj::qv]->array(mfi);
        auto qc_array    = mic_fab_vars[MicVar_SatAdj::qc]->array(mfi);
        auto tabs_array  = mic_fab_vars[MicVar_SatAdj::tabs]->array(mfi);
        auto theta_array = mic_fab_vars[MicVar_SatAdj::theta]->array(mfi);
        auto pres_array  = mic_fab_vars[MicVar_SatAdj::pres]->array(mfi);

#if 1
        // BEGIN ML SAT MODEL
        //=============================================================

        // set pytorch data type (default is float or torch::kFloat32)
        auto dtype0 = torch::kFloat64;

        // Tensor options for host only
        auto tensoropt = torch::TensorOptions().dtype(dtype0);

        // Auxiliary array for pytorch
        const long unsigned int nin  = 4; // T, P, Qv, Qc
        const long unsigned int nout = 3; // dT, dQv, dQc
        int ncell = tbx.numPts();
        Gpu::ManagedVector<Real> ML_aux(ncell*nin);
        Real* AMREX_RESTRICT ML_auxPtr = ML_aux.dataPtr();

        // Box attributes for index flattening
        const IntVect tbx_lo = tbx.smallEnd();
        const IntVect ntbox  = tbx.size();

        // Vector of inputs
        Vector<Array4<Real>> vec_ml_in = {tabs_array,
                                            qv_array,
                                            qc_array,
                                            pres_array};

        // Copy the ML inputs into auxiliary array
        for (int n(0); n<nin; ++n) {
            Array4<Real> data_arr = vec_ml_in[n];
            ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // n is indexing the variable 

                // Flatten indexing
                int ii = i - tbx_lo[0];
                int jj = j - tbx_lo[1];
                int index = jj*ntbox[0] + ii;
                int kk = k - tbx_lo[2];
                index += kk*ntbox[0]*ntbox[1];

                auto temporary = data_arr(i,j,k);

                // temperature scaling
                if (n == 0) {
                    // Apply MinMax scaling fit to training data
                    temporary = temperature_scaler_input.apply_scaling(temporary);
                }

                // qv scaling
                if (n == 1) {
                    // log(x + 1)-transform
                    temporary = std::log10(temporary + 1);
                    // MinMax scaling fit to training data
                    temporary = qv_scaler_input.apply_scaling(temporary);
                }

                // qc scaling
                if (n == 2) {
                    // log(x + 1)-transform
                    temporary = std::log10(temporary + 1);
                    // MinMax scaling fit to training data
                    temporary = qc_scaler_input.apply_scaling(temporary);
                }

                // pressure scaling
                if (n == 3) {
                    // apply log-transform to pressure variable
                    temporary = std::log10(temporary);
                    // MinMax scaling fit to training data
                    temporary = pres_scaler_input.apply_scaling(temporary);
                }


                // array order is row-based [index][comp]
                ML_auxPtr[index*nin + n] = temporary;
            });
        } // n

        // Create torch tensor from array
        at::Tensor inputs_torch = torch::from_blob(ML_auxPtr, {ncell, nin}, tensoropt);

        // Evaluate torch model
        at::Tensor outputs_torch = SatAdj_ML.forward({inputs_torch}).toTensor();
        outputs_torch = outputs_torch.to(dtype0);

        // Get accessor to output tensor (read-only, 2D {flatten ijk & nvar})
        auto outputs_torch_acc = outputs_torch.accessor<Real,2>();

        // Use output to modify dycore variables
        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            // Flatten indexing
            int ii = i - tbx_lo[0];
            int jj = j - tbx_lo[1];
            int index = jj*ntbox[0] + ii;
            int kk = k - tbx_lo[2];
            index += kk*ntbox[0]*ntbox[1];

            // Conserve total moisture
            Real Qt = qv_array(i,j,k) + qc_array(i,j,k);

            // ML output is {T, qv, qc}
            Real T_out_scaled = static_cast<Real>(outputs_torch_acc[index][0]);
            Real qv_out_scaled = static_cast<Real>(outputs_torch_acc[index][1]);

            // Undo MinMax scaling for output variables
            // Scaler was fit to output variables in training data
            Real T_out = temperature_scaler_output.undo_scaling(T_out_scaled);
            Real qv_out = qv_scaler_output.undo_scaling(qv_out);

            // undo log-transform for qv
            qv_out = std::exp(qv_out) - 1;

            tabs_array(i,j,k) = T_out;
              qv_array(i,j,k) = qv_out;
              qc_array(i,j,k) = Qt - qv_out;
           theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), pres_array(i,j,k), rdOcp);

            // Clip
            qv_array(i,j,k) = std::max(0.0, qv_array(i,j,k));
            qc_array(i,j,k) = std::max(0.0, qc_array(i,j,k));

        });
#else
        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            //qc_array(i,j,k) = std::max(0.0, qc_array(i,j,k));

            //------- Evaporation/condensation
            Real qsat;
            erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsat);

            // There is enough moisture to drive to equilibrium
            if ((qv_array(i,j,k)+qc_array(i,j,k)) > qsat) {
                Real qvprev = qv_array(i,j,k);
                Real qcprev = qc_array(i,j,k);

                // clip qc but maintain total water
                if (qc_array(i,j,k) < 0) {
                    qv_array(i,j,k) += qc_array(i,j,k);
                    qc_array(i,j,k)  = 0.0;
                }

                // Update temperature
                tabs_array(i,j,k) = NewtonIterSat(i, j, k   ,
                                                  d_fac_cond, tabs_array, pres_array,
                                                  qv_array  , qc_array  );

                Real qsatnew;
                erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatnew);
                amrex::ignore_unused(qvprev);
                amrex::ignore_unused(qcprev);
                AMREX_ASSERT(std::abs(qv_array(i,j,k)-qsatnew) < 1e-14);
                AMREX_ASSERT(std::abs(qv_array(i,j,k)+qc_array(i,j,k)-qvprev-qcprev) < 1e-14);

                // Update theta (constant pressure)
                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);

            //
            // We cannot blindly relax to qsat, but we can convert qc/qi -> qv.
            // The concept here is that if we put all the moisture into qv and modify
            // the temperature, we can then check if qv > qsat occurs (for final T/P/qv).
            // If the reduction in T/qsat and increase in qv does trigger the
            // aforementioned condition, we can do Newton iteration to drive qv = qsat.
            //
            } else {
                // Changes in each component
                Real delta_qc = qc_array(i,j,k);

                // Partition the change in non-precipitating q
                qv_array(i,j,k) += qc_array(i,j,k);
                qc_array(i,j,k)  = 0.0;

                // Update temperature (endothermic since we evap/sublime)
                tabs_array(i,j,k) -= d_fac_cond * delta_qc;

                // Update theta
                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);

                // Verify assumption that qv > qsat does not occur
                erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsat);
                if (qv_array(i,j,k) > qsat) {
                    Real qvprev = qv_array(i,j,k);
                    Real qcprev = qc_array(i,j,k);
                    Real Tprev = tabs_array(i,j,k);

                    // Update temperature
                    tabs_array(i,j,k) = NewtonIterSat(i, j, k     ,
                                                      d_fac_cond  , tabs_array, pres_array,
                                                      qv_array    , qc_array  );

                    Real qsatnew;
                    erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatnew);
                    amrex::ignore_unused(qvprev);
                    amrex::ignore_unused(qcprev);
                    amrex::ignore_unused(Tprev);
                    AMREX_ASSERT(qv_array(i,j,k) < qvprev);
                    AMREX_ASSERT(qc_array(i,j,k) > qcprev);
                    AMREX_ASSERT(tabs_array(i,j,k) > Tprev);
                    AMREX_ASSERT(std::abs(qv_array(i,j,k)-qsatnew) < 1e-14);
                    AMREX_ASSERT(std::abs(qv_array(i,j,k)+qc_array(i,j,k)-qvprev-qcprev) < 1e-14);

                    // Update theta
                    theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);

                }
            }

        });
#endif
    } // mfi
}

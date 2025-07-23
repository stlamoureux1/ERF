#ifndef ERF_SATADJ_H
#define ERF_SATADJ_H

/*
 * Implementation saturation adjustment microphysics model
 * The model transports qv and qc and does Newton iterations
 * to complete condensation/evaporation.
 */

#include <string>
#include <vector>
#include <memory>

// ML model
#include <torch/script.h>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_GpuContainers.H>

#include "ERF_EOS.H"
#include "ERF_Constants.H"
#include "ERF_MicrophysicsUtils.H"
#include "ERF_IndexDefines.H"
#include "ERF_DataStruct.H"
#include "ERF_NullMoist.H"
#include "ERF_TileNoZ.H"

namespace MicVar_SatAdj {
   enum {
      // independent variables
      rho=0, // density
      theta, // liquid/ice water potential temperature
      tabs,  // temperature
      pres,  // pressure
      // non-precipitating vars
      qv,    // cloud vapor
      qc,   // cloud water
      NumVars
  };
}

// Reproduce functionality from sklearn.preprocessing.MinMaxScaler
// The fitted scaling parameters from the training run are hard-coded below.
struct MinMaxScaler {
    amrex::Real data_min;
    amrex::Real data_max;
    amrex::Real scale;
    amrex::Real min;

    // These are the scaling and unscaling methods for the general case
    // With the range [0,1] they reduce to usual linear scaling
    // It might be wise to use that instead.
    amrex::Real apply_scaling(amrex::Real unscaled_value) {
        return (unscaled_value - data_min) / (data_max - data_min);
    }

    amrex::Real undo_scaling(amrex::Real scaled_value) {
        return scaled_value * (data_max - data_min) + data_min;
    }
};

class SatAdj : public NullMoist {

    using FabPtr = std::shared_ptr<amrex::MultiFab>;

public:
    // constructor
    SatAdj () {}

    // destructor
    virtual ~SatAdj () = default;

    // cloud physics
    void AdvanceSatAdj (const SolverChoice& /*solverChoice*/);

    // Set up for first time
    void
    Define (SolverChoice& sc) override
    {
        m_fac_cond = lcond / sc.c_p;
        m_rdOcp    = sc.rdOcp;
    }

    // init
    void
    Init (const amrex::MultiFab& cons_in,
          const amrex::BoxArray& /*grids*/,
          const amrex::Geometry& geom,
          const amrex::Real& dt_advance,
          std::unique_ptr<amrex::MultiFab>& /*z_phys_nd*/,
          std::unique_ptr<amrex::MultiFab>& /*detJ_cc*/) override;

    // Copy state into micro vars
    void
    Copy_State_to_Micro (const amrex::MultiFab& cons_in) override;

    // Copy state into micro vars
    void
    Copy_Micro_to_State (amrex::MultiFab& cons_in) override;

    // update micro vars
    void
    Update_Micro_Vars (amrex::MultiFab& cons_in) override
    {
        this->Copy_State_to_Micro(cons_in);
    }

    // update state vars
    void
    Update_State_Vars (amrex::MultiFab& cons_in) override
    {
        this->Copy_Micro_to_State(cons_in);
    }

    // wrapper to advance micro vars
    void
    Advance (const amrex::Real& dt_advance,
             const SolverChoice& solverChoice) override
    {
        dt = dt_advance;

        this->AdvanceSatAdj(solverChoice);
    }

    amrex::MultiFab*
    Qmoist_Ptr (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < m_qmoist_size);
        return nullptr;
    }

    int
    Qmoist_Size () override { return SatAdj::m_qmoist_size; }

    int
    Qstate_Moist_Size () override { return SatAdj::n_qstate_moist_size; }

    void
    Qmoist_Restart_Vars (const SolverChoice& /*a_sc*/,
                         std::vector<int>& a_idx,
                         std::vector<std::string>& a_names) const override
    {
        a_idx.clear();
        a_names.clear();
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::Real
    NewtonIterSat (int& i, int& j, int& k,
                   const amrex::Real& fac_cond,
                   const amrex::Array4<amrex::Real>& tabs_array,
                   const amrex::Array4<amrex::Real>& pres_array,
                   const amrex::Array4<amrex::Real>& qv_array,
                   const amrex::Array4<amrex::Real>& qc_array)
    {
        // Solution tolerance
        amrex::Real tol = 1.0e-8;

        // Saturation moisture fractions
        amrex::Real qsat, dqsat;

        // Newton iteration vars
        int  niter;
        amrex::Real fff, dfff;
        amrex::Real dtabs, delta_qv;

        // Initial guess for temperature & pressure
        amrex::Real tabs = tabs_array(i,j,k);
        amrex::Real pres = pres_array(i,j,k);

        niter = 0;
        dtabs = 1;

        //==================================================
        // Newton iteration to qv=qsat (cloud phase only)
        //==================================================
        do {
            // Saturation moisture fractions
            erf_qsatw(tabs, pres, qsat);
            erf_dtqsatw(tabs, pres, dqsat);

            // Function for root finding:
            // 0 = -T_new + T_old + L_eff/C_p * (qv - qsat)
            fff   = -tabs + tabs_array(i,j,k) + fac_cond*(qv_array(i,j,k) - qsat);

            // Derivative of function (T_new iterated on)
            dfff  = -1.0 - fac_cond*dqsat;

            // Update the temperature
            dtabs = -fff/dfff;
            tabs += dtabs;

            // Update iteration
            niter = niter+1;
        } while(std::abs(dtabs) > tol && niter < 20);

        // Update qsat from last iteration (dq = dq/dt * dt)
        qsat += dqsat*dtabs;

        // Changes in each component
        delta_qv = qv_array(i,j,k) - qsat;

        // Partition the change in non-precipitating q
        qv_array(i,j,k)  = qsat;
        qc_array(i,j,k) += delta_qv;

        // Return to temperature
        return tabs;
    }

private:
    // Number of qmoist variables (no precipitating comps to accumulate)
    int m_qmoist_size = 0;

    // Number of qstate variables
    int n_qstate_moist_size = 2;

    // geometry
    amrex::Geometry m_geom;

    // timestep
    amrex::Real dt;

    // constants
    amrex::Real m_fac_cond;
    amrex::Real m_rdOcp ;

    // independent variables
    amrex::Array<FabPtr, MicVar_SatAdj::NumVars> mic_fab_vars;

    // ML model
    std::string model_filename = "/home/slamoureux/ERF_ML/SatAdj/NNModel/model.pt";
    torch::jit::script::Module SatAdj_ML;

    // hard-code scaler parameters for X and Y from last training run
    // NOTE: normalization is applied after log-transforms
    // log-transforms are applied to pressure, qv_in, qv_out, qc_in, qc_out
    MinMaxScaler temperature_scaler_input {
        data_min: 190.0015527625396,
        data_max: 322.9978361188041,
        scale: 0.0075190071088020154,
        min: -1.4286230259049566
    };

    MinMaxScaler qv_scaler_input {
        data_min: 1.0581849347204382e-11,
        data_max: 0.06805378571493718,
        scale: 14.694259689597278,
        min: -1.554924423040166e-10
    };

    MinMaxScaler qc_scaler_input {
        data_min: 1.5335639184360813e-07,
        data_max: 0.004987455427613876,
        scale: 200.50921033443063,
        min: -3.074936902829939e-05
    };

    MinMaxScaler pres_scaler_input {
        data_min: 3.0020239296741202,
        data_max: 4.999991929195109,
        scale: 0.5005085167729159,
        min: -1.5025385443579942
    };

    MinMaxScaler temperature_scaler_output {
        data_min: 190.00152870902016,
        data_max: 374.2928539946578,
        scale: 0.00542619137634436,
        min: -1.0309846565731304,
    };

    MinMaxScaler qv_scaler_output {
        data_min: 3.7155708745993287e-09,
        data_max: 0.058699716379879086,
        scale: 17.0358585180612,
        min: -6.329793973350308e-08,
    };

    MinMaxScaler qc_scaler_output {
        data_min: 0.0,
        data_max: 0.026549110225738238,
        scale: 37.66604573551932,
        min: 0.0
    };
};
#endif

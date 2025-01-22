def build_cvxpy(self, relu_bin_formulation, thermal_model: str = "nn",
                target_name="Zone Mean Air Temperature(t+1)"):
    """
    This function only writes the EMS of the building. The optimization problem is written with cvxpy.
    None of the parameters is given a value. They are created as cvxpy parameters but not given a value.
    :param relu_bin_formulation: the formulation of the optimization problem
                - "milp": the relu binaries are variables
                - "fixed_bin": the relu binaries are parameters
                - "lp": the relu binaries are continuous variables between 0 and 1
    :param thermal_model: a str indicating which model is used for the thermodynamics of the building
                - "rcmodel" for the rc model
                - "nn" for the nn model
    :return:
    """

    
    def rc_formulation():
        # new parameters
        rc_alpha = {bn: {zn: {zzn: cp.Parameter(name=f"rc_alpha_{zn}_x_{zzn}")
                                for zzn in S_zones[bn]}
                            for zn in S_zones[bn]}
                    for bn in S_buildings}
        rc_inv_R = {bn: {zn: cp.Parameter(name=f"rc_inv_R_{zn}") for zn in S_zones[bn]}
                    for bn in S_buildings}
        rc_inv_C = {bn: {zn: cp.Parameter(name=f"rc_inv_C_{zn}") for zn in S_zones[bn]}
                    for bn in S_buildings}
        rc_h_eff = {bn: {zn: cp.Parameter(name=f"rc_h_eff_{zn}") for zn in S_zones[bn]}
                    for bn in S_buildings}
        rc_c_eff = {bn: {zn: cp.Parameter(name=f"rc_c_eff_{zn}") for zn in S_zones[bn]}
                    for bn in S_buildings}
        # new variable for cooling and heating power (must be positive)
        p_cooling = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_cooling_{zn}", nonneg=True) for zn in S_zones[bn]}
                        for bn in S_buildings}
        p_heating = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_heating_{zn}", nonneg=True) for zn in S_zones[bn]}
                        for bn in S_buildings}
        # intermediary variables to keep the problem DPP
        q_heating = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_q_heating_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
        q_cooling = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_q_cooling_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
        t_amb_div_R_dpp = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_t_amb_div_R_{zn}") for zn in S_zones[bn]}
                            for bn in S_buildings}
        t_in_div_R_dpp = {bn: {zn: cp.Variable((D_S_ts,), name=f"rc_t_in_div_R_{zn}") for zn in S_zones[bn]}
                            for bn in S_buildings}
        t_amb_var_dpp = cp.Variable((D_S_ts,), name=f"rc_t_amb_var_dpp")
        # Constraints for the RC model
        C_rc = []
        for bn in S_buildings:
            for t_idx, t in enumerate(S_ts):
                for z, zn in enumerate(S_zones[bn]):
                    # intermediary power
                    C_rc.append(p_hvac[bn][zn][t_idx] == p_cooling[bn][zn][t_idx] + p_heating[bn][zn][t_idx])
                    ### To keep problem DPP
                    # q heating
                    C_rc.append(q_heating[bn][zn][t_idx] == rc_h_eff[bn][zn] * p_heating[bn][zn][t_idx])
                    # q cooling
                    C_rc.append(q_cooling[bn][zn][t_idx] == rc_c_eff[bn][zn] * p_cooling[bn][zn][t_idx])
                    # intermediary t_amb variables (which is obviously equal to the parameter)
                    C_rc.append(t_amb_var_dpp[t_idx] == t_amb[t_idx])
                    # intermediary variable: t_amb * 1/R (var * param)
                    C_rc.append(t_amb_div_R_dpp[bn][zn][t_idx] == t_amb_var_dpp[t_idx] * rc_inv_R[bn][zn])
                    # intermediary variable: t_in * 1/R (var * param)
                    C_rc.append(t_in_div_R_dpp[bn][zn][t_idx] == t_in[bn][zn][t_idx] * rc_inv_R[bn][zn])

                    ### RC model
                    tmp0 = sum(t_in[bn][zzn][t_idx] * rc_alpha[bn][zn][zzn] for zzn in S_zones[bn])
                    tmp1 = q_heating[bn][zn][t_idx] * rc_inv_C[bn][zn]
                    tmp2 = - q_cooling[bn][zn][t_idx] * rc_inv_C[bn][zn]
                    tmp3 = t_amb_div_R_dpp[bn][zn][t_idx] * rc_inv_C[bn][zn]
                    tmp4 = - t_in_div_R_dpp[bn][zn][t_idx] * rc_inv_C[bn][zn]
                    if target_name == "Zone Mean Air Temperature(t+1)":
                        C_rc.append(t_in[bn][zn][t_idx + 1] == tmp0 + tmp1 + tmp2 + tmp3 + tmp4)
                    elif target_name == "Delta Mean Air Temperature(t)":  # Delta T
                        C_rc.append(
                            t_in[bn][zn][t_idx + 1] == tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + t_in[bn][zn][t_idx])

        return {"Parameters": [rc_alpha, rc_h_eff, rc_c_eff, rc_inv_R, rc_inv_C],
                "Variables": [p_cooling, p_heating],
                "Constraints": [C_rc]}

    ####################
    #  Sets
    ####################
    # Sets common to the community
    S_ts = self.ts_ems[:-1]  # time steps (midnight not included)
    S_ts_mn = self.ts_ems  # time steps (midnight included)
    S_buildings = self.bldg_assets.index.to_list()  # set of building names
    # Sets for each building
    S_zones = dict()  # keys = building names, values = list of zone names
    S_acus = dict()  # keys = building names, values = list of ACU names
    for bn in S_buildings:  # bn is the building name
        # Zone and ACU names for the building bn
        S_zones[bn] = self.bldg_assets[bn].zones_df_no_plenum["name"].to_list()
        S_acus[bn] = self.bldg_assets[bn].ACUs.values()

    # zones served by an ACU
    S_acu_zones = dict()
    for bn in S_buildings:
        # groupby acu name, make a list of the zone name in each group,
        # transform the tuple (acu_name, list of zone names) to a dict
        zonesbyacu = self.bldg_assets[bn].zones_df_no_plenum.groupby("ACU")["name"].apply(list).to_dict()
        S_acu_zones[bn] = zonesbyacu

    # dimensions of the sets 
    D_S_ts = len(S_ts)
    D_S_ts_mn = len(S_ts_mn)
    D_S_buildings = len(S_buildings)
    D_S_zones = {bn: len(S_zones[bn]) for bn in S_buildings}
    D_S_acus = {bn: len(S_acus[bn]) for bn in S_buildings}

    ####################
    #  Parameters
    ####################
    # exogenous variables
    t_amb = cp.Parameter((D_S_ts,), name=f"t_amb")
    hours = cp.Parameter((D_S_ts,), name=f"hours")
    nd_load = {bn: cp.Parameter((D_S_ts,), name=f"nd_load") for bn in S_buildings}

    # constant parameters
    line_capacity = cp.Parameter((1,), name="line_capacity")
    demand_charge = cp.Parameter((1,), name="demand_charge")
    prices_import = cp.Parameter((D_S_ts,), name="prices_import")
    prices_export = cp.Parameter((D_S_ts,), name="prices_export")
    ACU_capacities = {bn: {acu: cp.Parameter((1,), name=f"ACU_capacity_{acu}") for acu in S_acus[bn]}
                        for bn in S_buildings}
    hvac_capacities = {bn: {zn: cp.Parameter((1,), name=f"hvac_capacity_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
    Tin0 = {bn: {zn: cp.Parameter((1,), name=f"Tin0_{zn}") for zn in S_zones[bn]} for bn in S_buildings}
    Tmin = {bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"Tmin_{zn}") for zn in S_zones[bn]} for bn in S_buildings}
    Tmax = {bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"Tmax_{zn}") for zn in S_zones[bn]} for bn in S_buildings}

    ####################
    #  Variables
    ####################
    # Community-level variables
    p_demand_max = cp.Variable(name="p_demand_max", nonneg=True)
    p_import = cp.Variable((D_S_ts,), name="p_import", nonneg=True)  # no import decision for midnight next day
    p_export = cp.Variable((D_S_ts,), name="p_export", nonneg=True)  # no export decision for midnight next day
    # Buidling-level variables
    # -> cf. constraints
    # Zone-level variables
    # /!\ p_hvac is a dictionary of dictionary of cp.variables, it is not a cp.variable itself
    p_hvac = {bn: {zn: cp.Variable((D_S_ts,), name=f"p_hvac_{zn}", nonneg=True) for zn in S_zones[bn]} for bn in
                S_buildings}
    t_in = {bn: {zn: cp.Variable((D_S_ts_mn,), name=f"t_in_{zn}") for zn in S_zones[bn]} for bn in S_buildings}

    ####################
    #  Constraints
    ####################
    ### Community-level constraints
    # the community can not import more than the line capacity
    C_line_capacity = [p_demand_max <= line_capacity]
    # the peak demand is the maximum of the import minus the export (constraint for each ts)
    C_peak_demand = [p_demand_max >= p_import[ts_idx] - p_export[ts_idx] for ts_idx, _ in enumerate(S_ts)]
    # the power balance (constraint for each ts, sum over all buildings)
    C_power_balance = [p_import[ts_idx] - p_export[ts_idx] == sum(nd_load[bn][ts_idx] +
                                                                    sum(p_hvac[bn][zn][ts_idx] for zn in S_zones[bn])
                                                                    for bn in S_buildings)
                        for ts_idx, ts in enumerate(S_ts)]

    ### Building-level constraints
    # build the RC formulation
    thermal_model == "rcmodel":
        C_thermal_model = [c for c_l in rc_formulation()["Constraints"] for c in c_l]

    ### ACU level constraints
    C_acu_capacity = []
    for bn in S_buildings:
        for acu in S_acus[bn]:
            C_acu_capacity.extend([sum(p_hvac[bn][zn][ts_idx] for zn in S_acu_zones[bn][acu]) <=
                                    ACU_capacities[bn][acu] for ts_idx, _ in enumerate(S_ts)])

    ### Zone level constraints
    C_init_temp = []
    C_cooling_setpoint = []
    C_heating_setpoint = []
    C_hvac_capacity = []
    C_dummymodel = []
    for bn in S_buildings:
        for zn in S_zones[bn]:
            C_init_temp.append(t_in[bn][zn][0] == Tin0[bn][zn])
            C_cooling_setpoint.extend([t_in[bn][zn][0] <= Tin0[bn][zn]] +
                                        [t_in[bn][zn][ts_idx] <= Tmax[bn][zn][ts_idx]
                                        for ts_idx, ts in enumerate(S_ts_mn)][1:])
            C_heating_setpoint.extend([t_in[bn][zn][0] >= Tin0[bn][zn]] +
                                        [t_in[bn][zn][ts_idx] >= Tmin[bn][zn][ts_idx]
                                        for ts_idx, ts in enumerate(S_ts_mn)][1:])
            C_hvac_capacity.extend([p_hvac[bn][zn][ts_idx] <= hvac_capacities[bn][zn]
                                    for ts_idx, _ in enumerate(S_ts)])
            C_dummymodel.extend([t_in[bn][zn][ts_idx + 1] == 0.8 * t_in[bn][zn][ts_idx] + 4 * p_hvac[bn][zn][ts_idx]
                                    for ts_idx, _ in enumerate(S_ts)])

    constraints = (C_line_capacity + C_peak_demand + C_power_balance + C_acu_capacity + C_init_temp
                    # + C_cooling_setpoint + C_heating_setpoint
                    + C_hvac_capacity + C_thermal_model)  # + C_dummymodel

    ####################
    #  Objective
    ####################
    # The occupancy weights must be initialized in the problem otherwise it is not identified as
    # Disciplined Convex Programming (DCP) since the square of the temperature difference is convex
    # but the weight could be negative.
    occupancy_weights = {bn: {zn: self.bldg_assets[bn].zone_assets[zn].occupancy_weights
                                for zn in S_zones[bn]} for bn in S_buildings}

    # consigne de temperature: Ideally 21° constantly-
    temperature_rule = {
        bn: {zn: cp.Parameter((D_S_ts_mn,), name=f"temp_rule_{zn}")  # [21 for ts_idx, _ in enumerate(S_ts_mn)]
                for zn in S_zones[bn]} for bn in S_buildings}
    # the objective function
    # (Un)comment the last two lines to make the ridge regression and thus, the problem convex
    obj = cp.Minimize(demand_charge * p_demand_max + sum(
        prices_import[ts_idx] * p_import[ts_idx] - prices_export[ts_idx] * p_export[ts_idx]
        for ts_idx, _ in enumerate(S_ts)) * self.dt_ems
                        # decision variable regularization
                        # + 1e-3 * sum(p_import[ts_idx] ** 2 + p_export[ts_idx] ** 2  # for ridge regression (1)
                        #              for ts_idx, ts in enumerate(S_ts))  # for ridge regression (2)
                        # temperature regularization
                        + sum(occupancy_weights[bn][zn][ts_idx] * (
                                                    t_in[bn][zn][ts_idx] - temperature_rule[bn][zn][ts_idx]) ** 2
                            for bn in S_buildings
                            for zn in S_zones[bn] for ts_idx, _ in enumerate(S_ts_mn))
                        )

    ####################
    #  Solution
    ####################
    prob = cp.Problem(obj, constraints)
    print(f"Problem is DPP: {prob.is_dpp()}")

    return prob


def set_cvxpy_parameters(self, prob, thermal_model: str = "nn", weights=None, biases=None, relu_binaries=None):
    def update_setpoints(S_ts_mn):
        """
        Update the setpoints of the zones
        :param heating_setpoint: the heating setpoint Series (midnight included)
        :param cooling_setpoint: the cooling setpoint Series (midnight included)
        :return:
        """
        for b in self.bldg_assets:
            for zn in b.zones_df_no_plenum["name"]:
                Tmin0 = b.zone_assets[zn].Tmin.loc[S_ts_mn[0]]
                Tmax0 = b.zone_assets[zn].Tmax.loc[S_ts_mn[0]]
                # bar and restaurants at the bottom
                if "bot" in zn:
                    # 16° until 8am, 18° until 11am, 21° until midnight
                    heating_setpoint = np.array([Tmin0] + [16] * 8 + [18] * 3 + [21] * 13)
                    # 24° until 8am, after 23° until midnight
                    cooling_setpoint = np.array([Tmax0] + [24] * 8 + [23] * 3 + [23] * 13)
                # constrain the temperature in the middle a bit more
                elif "mid" in zn:
                    heating_setpoint = np.concatenate((Tmin0.reshape(1, ), np.clip(
                        b.zone_assets[zn].Tmin.loc[S_ts_mn[1:]].values, 17, 21)), axis=0)
                    cooling_setpoint = np.concatenate((Tmin0.reshape(1, ), np.clip(
                        b.zone_assets[zn].Tmax.loc[S_ts_mn[1:]].values - 1, 21, 24)), axis=0)
                # housing in the top floor
                elif "top" in zn:
                    # 17° until 6am, 21° until 8am, 16° until 5pm, 21° until 10pm, 17° until midnight
                    heating_setpoint = np.array([Tmin0] + [17] * 6 + [21] * 2 + [16] * 9 + [21] * 5 + [17] * 2)
                    # 23° until 8am, 26° until 5pm, 23° until midnight
                    cooling_setpoint = np.array([Tmax0] + [23] * 8 + [26] * 9 + [23] * 7)
                else:
                    raise ValueError(f"Zone '{zn}' not recognized")

                # b.zone_assets[zn].Tmin.loc[S_ts_mn] = heating_setpoint
                # b.zone_assets[zn].Tmax.loc[S_ts_mn] = cooling_setpoint
                b.zone_assets[zn].Tmin.loc[S_ts_mn[1:5]] = b.zone_assets[zn].Tmin.loc[S_ts_mn[1:5]] - 0.5
                b.zone_assets[zn].Tmin.loc[S_ts_mn[5:]] = b.zone_assets[zn].Tmin.loc[S_ts_mn[5:]] + 0.2
                b.zone_assets[zn].Tmax.loc[S_ts_mn[1:5]] = b.zone_assets[zn].Tmax.loc[S_ts_mn[1:5]] + 0.5
                b.zone_assets[zn].Tmax.loc[S_ts_mn[5:]] = b.zone_assets[zn].Tmax.loc[S_ts_mn[5:]] - 0.2

    S_ts = self.ts_ems[:-1]  # time steps (midnight not included)
    S_ts_mn = self.ts_ems  # time steps (midnight included)
    S_buildings = self.bldg_assets.index.to_list()  # set of building names
    S_zones = {bn: self.bldg_assets[bn].zones_df_no_plenum["name"].to_list() for bn in S_buildings}
    S_ACUs = {bn: self.bldg_assets[bn].ACUs.values() for bn in S_buildings}
    D_S_ts = len(S_ts)

    line_capacity = prob.param_dict.get("line_capacity")
    demand_charge = prob.param_dict.get("demand_charge")
    prices_import = prob.param_dict.get("prices_import")
    prices_export = prob.param_dict.get("prices_export")
    t_amb = prob.param_dict.get("t_amb")
    hours = prob.param_dict.get("hours")
    nd_load = {bn: prob.param_dict.get(f"nd_load") for bn in S_buildings}
    ACU_capacities = {bn: {acu: prob.param_dict.get(f"ACU_capacity_{acu}") for acu in S_ACUs[bn]}
                        for bn in S_buildings}
    hvac_capacities = {bn: {zn: prob.param_dict.get(f"hvac_capacity_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}
    Tin0 = {bn: {zn: prob.param_dict.get(f"Tin0_{zn}") for zn in S_zones[bn]}
            for bn in S_buildings}
    Tmin = {bn: {zn: prob.param_dict.get(f"Tmin_{zn}") for zn in S_zones[bn]}
            for bn in S_buildings}
    Tmax = {bn: {zn: prob.param_dict.get(f"Tmax_{zn}") for zn in S_zones[bn]}
            for bn in S_buildings}
    temperature_rule = {bn: {zn: prob.param_dict.get(f"temp_rule_{zn}") for zn in S_zones[bn]}
                        for bn in S_buildings}

    # update the setpoint values
    # update_setpoints(S_ts_mn)

    # set the values of the parameters only if not performing the cvxpylayer formulation
    line_capacity.value = np.array(self.market.line_capacity).reshape(1, )
    demand_charge.value = np.array(self.market.demand_charge).reshape(1, )
    prices_import.value = self.market.prices_import.loc[S_ts].values
    prices_export.value = self.market.prices_export.loc[S_ts].values
    bn = ""
    for b in self.bldg_assets:
        bn = b.name
        # To better see the scheduling, the base load is assumed to be 0 at the moment
        # nd_load[bn].value = self.bldg_assets[bn].nd_load.loc[S_ts].values
        nd_load[bn].value = np.zeros(D_S_ts)
        for acu in S_ACUs[bn]:
            ACU_capacities[bn][acu].value = np.array(self.bldg_assets[bn].ACU_capacity[acu]).reshape(1, )
        for zn in S_zones[bn]:
            hvac_capacities[bn][zn].value = np.array(
                self.bldg_assets[bn].zone_assets[zn].hvac_capacity).reshape(1, )
            Tin0[bn][zn].value = np.array(self.bldg_assets[bn].zone_assets[zn].Tin0).reshape(1, )
            # Tmin and Tmax might be none because of the temperature regularization (rather than constraint)
            if Tmin[bn][zn] is not None:
                Tmin[bn][zn].value = self.bldg_assets[bn].zone_assets[zn].Tmin.loc[S_ts_mn].values
            if Tmax[bn][zn] is not None:
                Tmax[bn][zn].value = self.bldg_assets[bn].zone_assets[zn].Tmax.loc[S_ts_mn].values
            # obj function parameters
            temperature_rule[bn][zn].value = np.array([21] * len(S_ts_mn))
    # same ambient temperature for all buildings
    if t_amb is not None:
        t_amb.value = self.bldg_assets[bn].simulation.loc[
            S_ts, "Site Outdoor Air Drybulb Temperature,ENVIRONMENT"].values
    if hours is not None:
        hours.value = self.bldg_assets[bn].simulation.loc[S_ts, "Hour"].values

    if thermal_model == "nn":
        # set the NN parameters
        for b in self.bldg_assets:
            # read the nn from onnx
            layers_onnx = list(b.nn.layers)
            bn = b.name
            # Set the values of the NN parameters
            for l, layer_onnx in enumerate(layers_onnx):
                if not isinstance(layer_onnx, InputLayer):
                    prob.param_dict.get(f"nn_layer_{l}_weights").value = weights[bn][l].detach().cpu().numpy()
                    prob.param_dict.get(f"nn_layer_{l}_biases").value = biases[bn][l].detach().cpu().numpy()
                # set the value of the binaries (if they are to be fixed)
                if layer_onnx.activation == "relu" and relu_binaries is not None:
                    for o in [i[0] for i in layer_onnx.output_indexes]:
                        prob.param_dict.get(f"nn_layer_{l}_binary_dpp_{o}").value = relu_binaries[bn][l][o]
        # set the value of the bounds
        self.set_zhat_and_z_bounds(prob)
    # for the rcmodel
    elif thermal_model == "rcmodel":
        for bn in S_buildings:
            for zn in S_zones[bn]:
                # WARNING: the numpy array and the original tensor share the same memory
                # check numpy in torch doc for more information
                # try:
                #     print(f"alpha keys: {list(weights[bn][zn]['alpha'].keys())}")  # DEBUG
                # except KeyError:
                #     print(f"weights[bn]: {weights[bn]}")
                param_dict = prob.param_dict
                for zzn in S_zones[bn]:
                    # print(f"alpha parameter: {prob.param_dict.get(f'rc_alpha_{zn}_x_{zzn}')}")
                    alpha = weights[bn][zn]["alpha"][zzn].detach().numpy()
                    param_dict.get(f"rc_alpha_{zn}_x_{zzn}").value = alpha
                    v = alpha
                    if np.isnan(v).any():
                        debug = 1
                        print(f"variable is nan: {v}")
                    elif np.isinf(v).any():
                        debug = 1
                        print(f"variable is inf: {v}")
                    elif 0 < v < 1e-5:
                        debug = 1
                        print(f"variable is too small: {v}")
                    elif v > 1e5:
                        debug = 1
                        print(f"variable is too large: {v}")
                h_eff = weights[bn][zn]["h_eff"].detach().numpy()
                c_eff = weights[bn][zn]["c_eff"].detach().numpy()
                R = weights[bn][zn]["R"].detach().item()
                C = weights[bn][zn]["C"].detach().item()
                beta_h = h_eff / C
                beta_c = c_eff / C
                gamma = 1 / (R * C)
                for v in [h_eff, c_eff, R, C]:
                    if np.isnan(v).any():
                        debug = 1
                        print(f"variable is nan: {v}")
                    elif np.isinf(v).any():
                        debug = 1
                        print(f"variable is inf: {v}")
                    elif 0 < v < 1e-5:
                        debug = 1
                        print(f"variable is too small: {v}")
                    elif v > 1e5:
                        debug = 1
                        print(f"variable is too large: {v}")
                param_dict.get(f"rc_inv_R_{zn}").value = 1 / R
                param_dict.get(f"rc_inv_C_{zn}").value = 1 / C
                param_dict.get(f"rc_h_eff_{zn}").value = h_eff
                param_dict.get(f"rc_c_eff_{zn}").value = c_eff

def solve_cvxpy(self, prob):
    # print(cp.installed_solvers())  # to check the installed solvers
    # # Print the constraints
    # print(prob.objective)
    # # Print the constraints
    # for constraint in prob.constraints:
    #     print(constraint)
    solver_engine = cp.GUROBI
    solver_parameters = {'verbose': False, 'TimeLimit': 60, 'MIPGap': 0.01, 'Threads': 10}
    try:
        prob.solve(solver=solver_engine, **solver_parameters)
        status = prob.status
    except cp.error.SolverError:
        for constraint in prob.constraints:
            print(constraint)
        solver_engine = cp.MOSEK
        solver_parameters = {'verbose': True, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
                                                                'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
                                                                'MSK_IPAR_NUM_THREADS': 10}}
        prob.solve(solver=solver_engine, **solver_parameters)
        stop = 1
    print("status:", prob.status)
    ### analyze the solution and save the results
    # if the solver returned an inaccurate, solve again with a new solver
    if prob.status == "optimal_inaccurate":
        # solver_engine = cp.MOSEK
        # solver_parameters = {'verbose': False, "mosek_params": {'MSK_DPAR_OPTIMIZER_MAX_TIME': 60,
        #                                                         'MSK_DPAR_MIO_TOL_REL_GAP': 0.01,
        #                                                         'MSK_IPAR_NUM_THREADS': 10}}
        # prob.solve(solver=solver_engine, **solver_parameters)
        solver_engine = cp.GUROBI
        solver_parameters = {'verbose': False, 'TimeLimit': 60, 'MIPGap': 0.01, 'Threads': 10}
        prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
        print("status:", prob.status)
    # if no incumbent has been found within the time limit, multiply the time limit by 10 and resolve.
    while prob.status == "infeasible_inaccurate":
        print(f"Problem not solved after {solver_parameters['TimeLimit']}s. Time limit increased to "
                f"{solver_parameters['TimeLimit'] * 10}s.")
        solver_parameters["TimeLimit"] *= 10
        solver_parameters["MIPGap"] *= 10 if solver_parameters["MIPGap"] < 1 else 1
        prob.solve(reoptimize=True, **solver_parameters)
        print("status:", prob.status)
    # if there is a solution
    if prob.solver_stats.extra_stats.SolCount >= 1:
        obj_val = prob.value
        try:
            mip_gap = prob.solver_stats.extra_stats.MIPGap
        except AttributeError:
            mip_gap = None
        solving_time = prob.solver_stats.solve_time
        # if the solution is optimal
        if prob.status == "optimal":
            print('* OPTIMISATION SUCCESSFUL *')
            print(f'Solving time = {solving_time:.2f} s')
            print("Solution value:\t", obj_val)
        # if the solution is suboptimal (or not guaranteed), stop due to time budget reached
        elif prob.status == "user_limit":
            print('* OPTIMISATION MET TERMINATION CRITERIA *')
            print(f'MIP gap = {mip_gap:.2%}')
            print("Solution value:\t", obj_val)

        p_import_val = prob.var_dict["p_import"].value.reshape(-1, 1)
        p_export_val = prob.var_dict["p_export"].value.reshape(-1, 1)
        if ((p_import_val * p_export_val).round(3) != 0).any():
            while True:
                x = 0
            print(f"p_import_val: {p_import_val}")
            print(f"p_export_val: {p_export_val}")

            # raise ValueError('Warning: Import and export powers cannot be non-zero at the same time')
        p_demand_max_val = prob.var_dict["p_demand_max"].value
        p_demand_max_vec = np.full((self.nb_ts_ems - 1, 1), p_demand_max_val)

        # store the results
        for b in self.bldg_assets:
            bn = b.name
            variables = []
            for z in b.zone_assets:
                if z.controlled:
                    zn = z.name
                    t_in_val = prob.var_dict[f"t_in_{zn}"].value.reshape(-1, 1)
                    p_hvac_val = prob.var_dict[f"p_hvac_{zn}"].value.reshape(-1, 1)
                    p_hvac_val = np.concatenate((p_hvac_val, p_hvac_val[-1, :].reshape((1, 1))), axis=0)
                    variables.extend([t_in_val.flatten(), p_hvac_val.flatten()])
                    z.expected_results = pd.DataFrame(data=np.concatenate((t_in_val, p_hvac_val), axis=1),
                                                        index=self.ts_ems, columns=['Tin', 'P_hvac'])
            col_names = pd.MultiIndex.from_product([b.zones_df_no_plenum["name"], ['Tin', 'P_hvac']],
                                                    names=['zone', 'variable'])
            b.expected_results = pd.DataFrame(np.array(variables).T, index=self.ts_ems, columns=col_names)

            ### Save the results of the MILP
            self.solution = dict(zip(prob.var_dict.keys(), prob.solution.primal_vars.values()))
            self.expected_results = pd.DataFrame(data=np.concatenate((p_demand_max_vec, p_import_val, p_export_val),
                                                                        axis=1),
                                                    index=self.ts_ems[:-1],
                                                    columns=['P_demand_max', 'P_import', 'P_export'])
            self.solver_status = {'obj_val': obj_val,
                                    'solver_engine': solver_engine,
                                    'termination_condition': prob.status,
                                    'options': prob.solver_stats.extra_stats.Params,
                                    'mip_gap': mip_gap,
                                    'solving_time': solving_time,
                                    'status': prob.solver_stats.extra_stats,
                                    }

    # If a solution was not found, find if it is infeasible or unbounded
    else:
        solver_parameters["verbose"] = True
        prob.solve(reoptimize=True, solver=solver_engine, **solver_parameters)
        self.solution, self.expected_results = None, None
        self.solver_status = {'obj_val': None,
                                'solver_engine': solver_engine,
                                'termination_condition': prob.status,
                                'options': prob.solver_stats.extra_stats.Params,
                                'mip_gap': None,
                                'solving_time': 0,
                                'status': prob.solver_stats.extra_stats,
                                }

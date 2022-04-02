import cplex
import numpy as np
import math
import time
import copy
import generate_data


def bcps(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij[i],
                                        types=['B']*num_buyers))
    eta = list(cpx.variables.add(obj=[-x for x in si],
                                 lb=[0]*num_suppliers,
                                 ub=[1]*num_suppliers,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ind, val=[1.0] * num_parking_zones)],
            senses=["L"],
            rhs=[1.0])
    for i in range(num_parking_zones):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ebs[i] + eta, val=[1]*num_buyers + [-x for x in xij[i]])],
            senses=["E"],
            rhs=[0])
    cpx.objective.set_quadratic_coefficients(gcs)

    # solve
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    output = []
    obj = cpx.solution.get_objective_value()
    values = cpx.solution.get_values()
    ebs_b = np.array(values[:num_buyers*num_parking_zones])\
        .reshape(num_parking_zones, num_buyers)
    eta_s = np.array(values[num_buyers*num_parking_zones:])
    bj_r = np.floor(np.tile(np.array(bj), (num_parking_zones, 1)))
    si_r = np.ceil(si)
    ext_num = ebs_b.sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def bcps_s1(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, padding, si_coff, padding_num):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij[i],
                                        types=['B']*num_buyers))
    eta = list(cpx.variables.add(obj=[-x for x in si_coff],
                                 lb=[0]*num_suppliers,
                                 ub=[1]*num_suppliers,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ind, val=[1.0] * num_parking_zones)],
            senses=["L"],
            rhs=[1.0])
    for i in range(num_parking_zones):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ebs[i] + eta, val=[1]*num_buyers + [-x for x in xij[i]])],
            senses=["E"],
            rhs=[-1*padding[i]])
    cpx.objective.set_quadratic_coefficients(gcs)

    # solve
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    output = []
    obj = cpx.solution.get_objective_value()
    values = cpx.solution.get_values()
    ebs_b = np.array(values[:num_buyers*num_parking_zones])\
        .reshape(num_parking_zones, num_buyers)
    eta_s = np.array(values[num_buyers*num_parking_zones:])
    bj_r = np.floor(np.tile(np.array(bj), (num_parking_zones, 1)))
    si_r = np.ceil(si)
    ext_num = ebs_b.sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj - padding_num)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def bcps_s2(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, padding, si_coff, padding_num, b_id):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    types = ['B']*num_buyers
    zij_n = copy.deepcopy(zij)
    types[b_id] = 'I'
    for i in range(num_parking_zones):
        if zij_n[i][b_id] == 1:
            zij_n[i][b_id] = 2
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij_n[i],
                                        types=types))

    eta = list(cpx.variables.add(obj=[-x for x in si_coff],
                                 lb=[0]*num_suppliers,
                                 ub=[1]*num_suppliers,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        if i == b_id:
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=ind, val=[1.0] * num_parking_zones)],
                senses=["L"],
                rhs=[2.0])
        else:
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=ind, val=[1.0] * num_parking_zones)],
                senses=["L"],
                rhs=[1.0])
    for i in range(num_parking_zones):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ebs[i] + eta, val=[1]*num_buyers + [-x for x in xij[i]])],
            senses=["E"],
            rhs=[-1*padding[i]])
    cpx.objective.set_quadratic_coefficients(gcs)
    # solve
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    output = []
    obj = cpx.solution.get_objective_value()
    values = cpx.solution.get_values()
    ebs_b = np.array(values[:num_buyers*num_parking_zones])\
        .reshape(num_parking_zones, num_buyers)
    eta_s = np.array(values[num_buyers*num_parking_zones:])
    bj_r = np.floor(np.tile(np.array(bj), (num_parking_zones, 1)))
    si_r = np.ceil(si)
    ext_num = ebs_b.sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj - padding_num)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def bcps_s3(num_suppliers, num_buyers, num_parking_zones, zij_n, xij, gcs, g_c, bj, si):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij_n[i],
                                        types=['B']*num_buyers))

    eta = list(cpx.variables.add(obj=[-x for x in si],
                                 lb=[0]*num_suppliers,
                                 ub=[1]*num_suppliers,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ind, val=[1.0] * num_parking_zones)],
            senses=["L"],
            rhs=[1])
    for i in range(num_parking_zones):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ebs[i] + eta, val=[1]*num_buyers + [-x for x in xij[i]])],
            senses=["E"],
            rhs=[0])
    cpx.objective.set_quadratic_coefficients(gcs)
    # solve
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    output = []
    obj = cpx.solution.get_objective_value()
    values = cpx.solution.get_values()
    ebs_b = np.array(values[:num_buyers*num_parking_zones])\
        .reshape(num_parking_zones, num_buyers)
    eta_s = np.array(values[num_buyers*num_parking_zones:])
    bj_r = np.floor(np.tile(np.array(bj), (num_parking_zones, 1)))
    si_r = np.ceil(si)
    ext_num = ebs_b.sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[num_buyers*num_parking_zones:]
    return output


def bcps_s4(num_suppliers, num_buyers, num_parking_zones, zij_n, xij, gcs, g_c, bj, si, s_id):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij_n[i],
                                        types=['B']*num_buyers))
    ub_s = [1]*num_suppliers
    ub_s[s_id] = 0
    eta = list(cpx.variables.add(obj=[-x for x in si],
                                 lb=[0]*num_suppliers,
                                 ub=ub_s,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ind, val=[1.0] * num_parking_zones)],
            senses=["L"],
            rhs=[1.0])
    for i in range(num_parking_zones):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=ebs[i] + eta, val=[1]*num_buyers + [-x for x in xij[i]])],
            senses=["E"],
            rhs=[0])
    cpx.objective.set_quadratic_coefficients(gcs)

    # solve
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    output = []
    obj = cpx.solution.get_objective_value()
    values = cpx.solution.get_values()
    ebs_b = np.array(values[:num_buyers*num_parking_zones])\
        .reshape(num_parking_zones, num_buyers)
    eta_s = np.array(values[num_buyers*num_parking_zones:])
    bj_r = np.floor(np.tile(np.array(bj), (num_parking_zones, 1)))
    si_r = np.ceil(si)
    ext_num = ebs_b.sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def run_bcps(num_suppliers, num_buyers, num_parking_zones, data, padding):
    # utility_s, utility_b, pay, ext_cost, max sw, sw, time, efficency, trade num
    bcps_output = [0] * 9
    bcps_start_time = time.time()
    # esc
    bcps_r1 = bcps_s1(num_suppliers,
                      num_buyers,
                      num_parking_zones,
                      data.zij, data.xij, data.gcs, data.g_c,
                      data.bj, data.si, data.padding_v, data.si_coff,
                      data.padding_num)
    # calculate price for buyers
    zij_n = copy.deepcopy(data.zij)
    ebs_b_org = np.array(bcps_r1[2:]).reshape(num_parking_zones, num_buyers)
    ebs_b = ebs_b_org.sum(axis=0)
    for i in range(num_buyers):
        if ebs_b[i] > 0.99:
            bcps_r2 = bcps_s2(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              data.zij, data.xij, data.gcs, data.g_c,
                              data.bj, data.si, data.padding_v, data.si_coff,
                              data.padding_num, i)
            if bcps_r1[0] < bcps_r2[0]:
                bcps_output[1] += bcps_r2[1] - bcps_r1[1]
                b_price = math.floor(data.bj[i]) - (bcps_r2[1] - bcps_r1[1])
                bcps_output[2] += b_price
            else:
                for j in range(num_parking_zones):
                    zij_n[j][i] = 0
        else:
            for j in range(num_parking_zones):
                zij_n[j][i] = 0
    # allocation
    bcps_r3 = bcps_s3(num_suppliers,
                      num_buyers,
                      num_parking_zones,
                      zij_n, data.xij, data.gcs, data.g_c,
                      data.bj, data.si)
    # calculate price for suppliers
    eta_s = bcps_r3[2:]
    for i in range(num_suppliers):
        if eta_s[i] > 0.99:
            bcps_r4 = bcps_s4(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              zij_n, data.xij, data.gcs, data.g_c,
                              data.bj, data.si, i)
            bcps_output[0] += bcps_r3[1] - bcps_r4[1]
            s_price = math.ceil(data.si[i]) + (bcps_r3[1] - bcps_r4[1])
            bcps_output[2] -= s_price
    bcpv_r = bcps(num_suppliers,
                  num_buyers,
                  num_parking_zones,
                  data.zij, data.xij, data.gcs, data.g_c,
                  data.bj, data.si)
    ext_num = np.array(eta_s).reshape((num_parking_zones, -1)).sum(axis=1)
    bcps_output[3] = (ext_num*ext_num*data.g_c_2).sum()
    bcps_output[4] = bcpv_r[1]
    bcps_output[5] = bcps_r3[1]
    bcps_output[6] = time.time() - bcps_start_time
    bcps_output[7] = bcps_output[5] * 1.0 / bcps_output[4]
    bcps_output[8] = sum(eta_s)
    return bcps_output


if __name__ == "__main__":
    num_suppliers = 4
    num_buyers = 6
    num_parking_zones = 2
    padding = 1

    data = generate_data.GenData()
    data.generate_data(True, True)
    data.zij = [[1, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 1]]
    data.bj = [44, 45, 43, 44, 45, 43]
    data.si = [40, 42, 40, 42]
    data.xij = [[1, 1, 0, 0], [0, 0, 1, 1]]
    data.g_c = [1, 1]
    data.gcs = [(12, 12, -2), (12, 13, -2), (13, 13, -2), (14, 14, -2),
            (14, 15, -2), (15, 15, -2)]
    data.padding_num = 2
    data.padding_v = [1, 1]
    data.si_coff = [38, 40, 38, 40]
    gcs_b_new = []
    current_state = 0
    for k in range(num_parking_zones):
        for i in range(current_state, current_state + num_buyers):
            for j in range(i, current_state + num_buyers):
                gcs_b_new.append((i, j, -2))
        current_state += num_buyers
    data.gcs_b = gcs_b_new
    result = run_bcps(num_suppliers, num_buyers, num_parking_zones, data, padding)
    print(1)

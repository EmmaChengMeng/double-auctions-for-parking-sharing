import cplex
import numpy as np
import math
import time
import copy
import generate_data

def scpv(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si):
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
    ext_num = eta_s.reshape((num_parking_zones, -1)).sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def scpv_s1(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, padding):
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
            rhs=[padding[i]])
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
    ext_num = eta_s.reshape((num_parking_zones, -1)).sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[num_buyers*num_parking_zones:]
    return output


def scpv_s2(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, padding, s_id):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    # varaibles
    ebs = [None] * num_parking_zones
    for i in range(num_parking_zones):
        ebs[i] = list(cpx.variables.add(obj=bj,
                                        lb=[0]*num_buyers,
                                        ub=zij[i],
                                        types=['B']*num_buyers))
    ub = [1] * num_suppliers
    ub[s_id] = 0
    eta = list(cpx.variables.add(obj=[-x for x in si],
                                 lb=[0]*num_suppliers,
                                 ub=ub,
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
            rhs=[padding[i]])
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
    ext_num = eta_s.reshape((num_parking_zones, -1)).sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[num_buyers*num_parking_zones:]
    return output


def scpv_s3(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, ub_s):
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
            rhs=[0.0])
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
    ext_num = eta_s.reshape((num_parking_zones, -1)).sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values
    return output


def scpv_s4(num_suppliers, num_buyers, num_parking_zones, zij, xij, gcs, g_c, bj, si, ub_s, b_id):
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
                                 ub=ub_s,
                                 types=['B']*num_suppliers))
    # constraint
    for i in range(num_buyers):
        ind = [ebs[j][i] for j in range(num_parking_zones)]
        if i == b_id:
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=ind, val=[1.0] * num_parking_zones)],
                senses=["L"],
                rhs=[0])
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
            rhs=[0.0])
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
    ext_num = eta_s.reshape((num_parking_zones, -1)).sum(axis=1)
    ext_cost = (ext_num*ext_num*g_c).sum()
    real_obj = (bj_r*ebs_b).sum() - (si_r*eta_s).sum() - ext_cost
    output.append(obj)
    output.append(real_obj)
    output += values[:num_buyers*num_parking_zones]
    return output


def run_scpv(num_suppliers, num_buyers, num_parking_zones, data, padding):
    # utility_s, utility_b, pay, ext_cost, max sw, sw, time, efficency, trade_num
    scpv_output = [0] * 9
    scpv_start_time = time.time()
    # scpv
    scpv_r1 = scpv_s1(num_suppliers,
                      num_buyers,
                      num_parking_zones,
                      data.zij, data.xij, data.gcs,
                      data.g_c, data.bj, data.si, data.padding_v2)
    # calculate price for suppliers
    ub_s = [0] * num_suppliers
    s_utility_dict = {}
    s_dict = {}
    for i in range(num_suppliers):
        if scpv_r1[i + 2] > 0.99:
            scpv_r2 = scpv_s2(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              data.zij, data.xij, data.gcs, data.g_c,
                              data.bj, data.si, data.padding_v2, i)
            scpv_output[0] += scpv_r1[1] - scpv_r2[1]
            s_price = math.ceil(data.si[i]) + scpv_r1[1] - scpv_r2[1]
            s_utility_dict[i] = scpv_r1[1] - scpv_r2[1]
            s_dict[i] = s_price
            scpv_output[2] -= s_price
            ub_s[i] = 1
    # allocation
    scpv_r3 = scpv_s3(num_suppliers,
                      num_buyers,
                      num_parking_zones,
                      data.zij, data.xij, data.gcs, data.g_c,
                      data.bj, data.si, ub_s)
    # calculate price for buyers
    ebs_b_org = np.array(scpv_r3[2:num_buyers*num_parking_zones+2])\
        .reshape(num_parking_zones, num_buyers)
    ebs_b = ebs_b_org.sum(axis=0)
    b_utility_dict = {}
    b_dict = {}
    for i in range(num_buyers):
        if ebs_b[i] > 0.99:
            scpv_r4 = scpv_s4(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              data.zij, data.xij, data.gcs, data.g_c,
                              data.bj, data.si, ub_s, i)
            scpv_output[1] += scpv_r3[1] - scpv_r4[1]
            b_price = math.floor(data.bj[i]) - (scpv_r3[1] - scpv_r4[1])
            b_utility_dict[i] = scpv_r3[1] - scpv_r4[1]
            b_dict[i] = b_price
            scpv_output[2] += b_price
    scpv_r = scpv(num_suppliers,
                  num_buyers,
                  num_parking_zones,
                  data.zij, data.xij, data.gcs, data.g_c,
                  data.bj, data.si)
    ext_num = np.array(scpv_r1[2:]).\
        reshape((num_parking_zones, -1)).sum(axis=1)
    scpv_output[3] = (ext_num*ext_num*data.g_c_2).sum()
    scpv_output[4] = scpv_r[1]
    scpv_output[5] = scpv_r3[1]
    scpv_output[6] = time.time() - scpv_start_time
    scpv_output[7] = scpv_output[5] * 1.0 / scpv_output[4]
    scpv_output[8] = sum(ebs_b)
    return scpv_output


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
    result = run_scpv(num_suppliers, num_buyers, num_parking_zones, data, padding)
    print(1)

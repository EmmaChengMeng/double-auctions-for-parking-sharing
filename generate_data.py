import random
import math
import numpy as np
from collections import Counter
import codecs
import copy
import cplex
import sc_pv_qp
import sc_ps_qp
import bc_pv_qp
import bc_ps_qp
import matplotlib.pyplot as plt
import pandas as pd


exp_name = 'rc_1.5'
max_parking_zones = 5
rc_num = 20
ob_num = 20
suppliers_per_rc = 20
buyers_per_ob = 20
p_avg = 50
p_sd = 5
cycle = 1
r_cluster = 2
walk_dis_constraint = 0.8
walk_dis_per_min = 0.08
num_suppliers = rc_num * suppliers_per_rc
num_buyers = ob_num * buyers_per_ob
num_parking_zones = rc_num
g_c_max = 6
g_c_min = 1
padding = 1
g_c_coff = 0.01


class GenData:
    def __init__(self):
        self.bj = [0] * num_buyers
        self.si = [0] * num_suppliers
        self.rc_loc = [None] * rc_num
        self.ob_loc = [None] * ob_num
        self.distance = [None] * ob_num
        self.rc_sets = [None] * ob_num
        self.zij = [None] * num_parking_zones
        self.xij = [None] * num_parking_zones
        self.si_org = [0] * num_suppliers
        self.bj_org = [0] * num_buyers
        self.g_c = [0] * num_parking_zones
        self.gcs = []
        self.gcs_b = []
        self.padding_v = [0] * num_parking_zones
        self.si_coff = [0] * num_suppliers
        self.padding_num = 0
        self.padding_v2 = []
        self.g_c_org = [0] * num_parking_zones
        self.g_c_2 = None

    def set_loc_data(self, rc_loc, ob_loc, distance, rc_sets):
        self.rc_loc = rc_loc
        self.ob_loc = ob_loc
        self.distance = distance
        self.rc_sets = rc_sets

    def set_org_price(self, si_price, bj_price):
        self.si_org = si_price
        self.bj_org = bj_price

    def set_g_c_org(self, g_c):
        self.g_c_org = g_c

    def set_g_c_2(self, g_c_2):
        self.g_c_2 = g_c_2

    def generate_data(self, gen_loc_flag, price_flag, g_c_flag):
        # generate the locations of residential communities
        if gen_loc_flag:
            for k in range(rc_num):
                length = random.random() * r_cluster
                angle = random.random() * 2 * math.pi
                x = length * math.cos(angle)
                y = length * math.sin(angle)
                self.rc_loc[k] = [x, y]

            for k in range(ob_num):
                length = random.random() * r_cluster
                angle = random.random() * 2 * math.pi
                x = length * math.cos(angle)
                y = length * math.sin(angle)
                self.ob_loc[k] = [x, y]

            for i in range(ob_num):
                self.distance[i] = []
                self.rc_sets[i] = {}
                for j in range(rc_num):
                    dis = math.sqrt(math.pow(self.ob_loc[i][0]-self.rc_loc[j][0], 2)
                                    + math.pow(self.ob_loc[i][1]-self.rc_loc[j][1], 2))
                    self.distance[i].append(dis)
                    if dis <= walk_dis_constraint:
                        self.rc_sets[i][j] = dis/walk_dis_per_min
        '''
        rc_sets_count = [len(rc_set) for rc_set in self.rc_sets]
        counts = dict(Counter(rc_sets_count))
        for i in range(11):
            if i in counts:
                print('{0}: {1}'.format(i, counts[i]))
        c20 = 0
        c30 = 0
        c30p = 0
        for k, v in counts.items():
            if 10 < k <= 20:
                c20 += v
            elif 20 < k <= 30:
                c30 += v
            elif k >= 30:
                c30p += v
        print('c20: {0}'.format(c20))
        print('c30: {0}'.format(c30))
        print('c30p: {0}'.format(c30p))
        '''
        # generate prices for suppliers
        s_count = 0
        if price_flag:
            raw_price = np.random.normal(p_avg, p_sd, num_suppliers)
            self.si_org = raw_price
        else:
            raw_price = self.si_org
        for i in range(num_suppliers):
            self.si[i] = round(raw_price[i]) - (s_count + 1)*10e-8
            s_count += 1

        for i in range(num_parking_zones):
            self.zij[i] = [0] * num_buyers

        for i in range(num_parking_zones):
            self.xij[i] = [0] * num_suppliers
            start_no = i * suppliers_per_rc
            for j in range(suppliers_per_rc):
                self.xij[i][start_no + j] = 1

        # generate prices for buyers
        p_count = 0
        if price_flag:
            raw_price = np.random.normal(p_avg, p_sd, num_buyers)
            self.bj_org = raw_price
        else:
            raw_price = self.bj_org
        for i in range(ob_num):
            pz_sets = self.select_parking_zone(self.rc_sets[i], max_parking_zones)
            if pz_sets:
                for j in range(buyers_per_ob):
                    b_count = i * buyers_per_ob + j
                    self.bj[b_count] = round(raw_price[b_count]) + (p_count + 1) * 10e-8
                    for ky, v in pz_sets.items():
                        self.zij[ky][b_count] = 1
                        self.padding_v[ky] = padding
                    p_count += 1
        self.padding_v2 = self.padding_v
        if g_c_flag:
            self.g_c_org = np.random.randint(g_c_min, g_c_max, num_parking_zones)
        self.g_c = g_c_coff * self.g_c_org
        self.g_c_2 = copy.deepcopy(self.g_c)
        current_state = num_buyers * num_parking_zones
        for k in range(num_parking_zones):
            coff = -2 * self.g_c[k]
            for i in range(current_state, current_state + suppliers_per_rc):
                for j in range(i, current_state + suppliers_per_rc):
                    self.gcs.append((i, j, coff))
            current_state = current_state + suppliers_per_rc
        current_state = 0
        for k in range(num_parking_zones):
            coff = -2 * self.g_c[k]
            for i in range(current_state, current_state + num_buyers):
                for j in range(i, current_state + num_buyers):
                    self.gcs_b.append((i, j, coff))
            current_state = current_state + num_buyers
        current_state = 0
        self.si_coff = copy.deepcopy(self.si)
        for k in range(num_parking_zones):
            if self.padding_v[k] != 0:
                self.padding_num += math.pow(self.padding_v[k], 2) * self.g_c[k]
                for i in range(current_state, current_state + suppliers_per_rc):
                    self.si_coff[i] -= 2 * self.g_c[k] * self.padding_v[k]
            current_state = current_state + suppliers_per_rc

    def select_parking_zone(self, pz_dict, pz_num):
        if len(pz_dict) <= pz_num:
            return pz_dict
        else:
            pz_list = [(ky, v) for ky, v in pz_dict.items()]
            output = sorted(pz_list, key=lambda x: x[1])
            return dict(output[:pz_num])

    def modify_padding_v(self, bj, zij):
        cpx = cplex.Cplex()
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        bj_len = len(bj)
        ebs = [None] * num_parking_zones
        for i in range(num_parking_zones):
            ebs[i] = list(cpx.variables.add(obj=bj,
                                            lb=[0] * bj_len,
                                            ub=zij[i],
                                            types=['B']*bj_len))
        # constraint
        for i in range(bj_len):
            ind = [ebs[j][i] for j in range(num_parking_zones)]
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=ind, val=[1.0] * num_parking_zones)],
                senses=["L"],
                rhs=[1.0])
        for i in range(num_parking_zones):
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=ebs[i], val=[1] * bj_len)],
                senses=["L"],
                rhs=[1.0])
        # solve
        cpx.solve()
        print("Solution status =", cpx.solution.get_status_string())
        print("Optimal value:", cpx.solution.get_objective_value())
        values = cpx.solution.get_values()
        p_v = np.array(values)\
            .reshape(num_parking_zones, bj_len).sum(axis=1)
        return p_v


'''
def main():
    sr_esc_result = []
    sr_ebc_result = []
    pbc_result = []
    pbc_p_result = []
    for cyl in range(cycle):
        data = GenData()
        data.generate_data(True, True)

        sr_esc_r = sr_esc.run_sr_esc(num_suppliers,
                                     num_buyers,
                                     num_parking_zones,
                                     buyers_per_ob, data)
        sr_esc_result.append(','.join([str(r) for r in sr_esc_r]))

        sr_ebc_r = sr_ebc.run_sr_ebc(num_suppliers,
                                     num_buyers,
                                     num_parking_zones,
                                     buyers_per_ob, data)
        sr_ebc_result.append(','.join([str(r) for r in sr_ebc_r]))
        
        pbc_r = pbc.run_pbc(num_suppliers,
                            num_buyers,
                            num_parking_zones,
                            buyers_per_ob, data, False)
        pbc_result.append(','.join([str(r) for r in pbc_r]))

        pbc_p_r = pbc.run_pbc(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              buyers_per_ob, data, True)
        pbc_p_result.append(','.join([str(r) for r in pbc_p_r]))

    with codecs.open('re_' + exp_name + '.csv', 'w', 'utf-8') as f:
        for re in sr_esc_result:
            f.write(str(re) + '\n')
        f.write('\n')
        f.write('\n')
        for re in sr_ebc_result:
            f.write(str(re) + '\n')
        f.write('\n')
        f.write('\n')
        for re in pbc_result:
            f.write(str(re) + '\n')
        f.write('\n')
        f.write('\n')
        for re in pbc_p_result:
            f.write(str(re) + '\n')
'''


def run_exp(num_suppliers, num_buyers, num_parking_zones, data_list, save_path):
    sc_pv_result = []
    bc_pv_result = []
    sc_ps_result = []
    bc_ps_result = []
    for cyl in range(cycle):
        data = data_list[cyl]
        if padding >= 1:
            sc_pv_r = sc_pv_qp.run_scpv(num_suppliers,
                                         num_buyers,
                                         num_parking_zones,
                                         data, padding)
            sc_pv_result.append(','.join([str(r) for r in sc_pv_r]))

            bc_pv_r = bc_pv_qp.run_bcpv(num_suppliers,
                                         num_buyers,
                                         num_parking_zones,
                                         data, padding)
            bc_pv_result.append(','.join([str(r) for r in bc_pv_r]))

        sc_ps_r = sc_ps_qp.run_scps(num_suppliers,
                            num_buyers,
                            num_parking_zones,
                            data, padding)
        sc_ps_result.append(','.join([str(r) for r in sc_ps_r]))

        bc_ps_r = bc_ps_qp.run_bcps(num_suppliers,
                              num_buyers,
                              num_parking_zones,
                              data, padding)
        bc_ps_result.append(','.join([str(r) for r in bc_ps_r]))

    with codecs.open(save_path + '_sc_pv.csv', 'a', 'utf-8') as f1:
        for re in sc_pv_result:
            f1.write(str(re) + '\n')
    with codecs.open(save_path + '_bc_pv.csv', 'a', 'utf-8') as f2:
        for re in bc_pv_result:
            f2.write(str(re) + '\n')
    with codecs.open(save_path + '_sc_ps.csv', 'a', 'utf-8') as f3:
        for re in sc_ps_result:
            f3.write(str(re) + '\n')
    with codecs.open(save_path + '_bc_ps.csv', 'a', 'utf-8') as f4:
        for re in bc_ps_result:
            f4.write(str(re) + '\n')


def main1():
    global suppliers_per_rc
    global buyers_per_ob
    global num_suppliers
    global num_buyers
    data_list_10_10 = []
    data_list_20_20 = []
    data_list_30_30 = []
    for cyl in range(cycle):
        data = GenData()
        data.generate_data(True, True, True)
        data_list_20_20.append(data)
    run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_20_20, '20-20')

    suppliers_per_rc = 10
    buyers_per_ob = 10
    num_suppliers = rc_num * suppliers_per_rc
    num_buyers = ob_num * buyers_per_ob
    for cyl in range(cycle):
        data = GenData()
        data.set_loc_data(data_list_20_20[cyl].rc_loc, data_list_20_20[cyl].ob_loc,
                          data_list_20_20[cyl].distance, data_list_20_20[cyl].rc_sets)
        data.set_g_c(data_list_20_20[cyl].g_c)
        data.generate_data(False, True, False)
        data_list_10_10.append(data)
    run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_10_10, '10-10')

    suppliers_per_rc = 30
    buyers_per_ob = 30
    num_suppliers = rc_num * suppliers_per_rc
    num_buyers = ob_num * buyers_per_ob
    for cyl in range(cycle):
        data = GenData()
        data.set_loc_data(data_list_20_20[cyl].rc_loc, data_list_20_20[cyl].ob_loc,
                          data_list_20_20[cyl].distance, data_list_20_20[cyl].rc_sets)
        data.set_g_c(data_list_20_20[cyl].g_c)
        data.generate_data(False, True, False)
        data_list_30_30.append(data)
    run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_30_30, '30_30')


def main2():
    global max_parking_zones
    for cyl2 in range(5):
        data_list_5 = []
        data_list_3 = []
        data_list_10 = []
        data_list_1 = []
        max_parking_zones = 5
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data_list_5.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_5, 'm_5')

        max_parking_zones = 3
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_5[cyl].rc_loc, data_list_5[cyl].ob_loc,
                              data_list_5[cyl].distance, data_list_5[cyl].rc_sets)
            data.set_org_price(data_list_5[cyl].si_org, data_list_5[cyl].bj_org)
            data.set_g_c_org(data_list_5[cyl].g_c_org)
            data.generate_data(False, False, False)
            data_list_3.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_3, 'm_3')

        max_parking_zones = 10
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_5[cyl].rc_loc, data_list_5[cyl].ob_loc,
                              data_list_5[cyl].distance, data_list_5[cyl].rc_sets)
            data.set_org_price(data_list_5[cyl].si_org, data_list_5[cyl].bj_org)
            data.set_g_c_org(data_list_5[cyl].g_c_org)
            data.generate_data(False, False, False)
            data_list_10.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_10, 'm_10')

        max_parking_zones = 1
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_5[cyl].rc_loc, data_list_5[cyl].ob_loc,
                              data_list_5[cyl].distance, data_list_5[cyl].rc_sets)
            data.set_org_price(data_list_5[cyl].si_org, data_list_5[cyl].bj_org)
            data.set_g_c_org(data_list_5[cyl].g_c_org)
            data.generate_data(False, False, False)
            data_list_1.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_1, 'm_1')


def main3():
    global suppliers_per_rc
    global buyers_per_ob
    global num_suppliers
    global num_buyers
    for cyl2 in range(50):
        data_list_1 = []
        data_list_3 = []
        data_list_20 = []
        data_list_30 = []
        data_list_50 = []
        suppliers_per_rc = 1
        num_suppliers = rc_num * suppliers_per_rc
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data_list_1.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_1, 's_1')

        suppliers_per_rc = 3
        num_suppliers = rc_num * suppliers_per_rc
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_3.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_3, 's_3')

        suppliers_per_rc = 20
        num_suppliers = rc_num * suppliers_per_rc
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_20.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_20, 's_20')

        suppliers_per_rc = 30
        num_suppliers = rc_num * suppliers_per_rc
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_30.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_30, 's_30')

        suppliers_per_rc = 50
        num_suppliers = rc_num * suppliers_per_rc
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_50.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_50, 's_50')


def main4():
    global suppliers_per_rc
    global buyers_per_ob
    global num_suppliers
    global num_buyers
    for cyl2 in range(3):
        data_list_1 = []
        data_list_3 = []
        data_list_20 = []
        data_list_30 = []
        data_list_50 = []
        buyers_per_ob = 1
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data.padding_v2 = data.modify_padding_v(data.bj, data.zij)
            data_list_1.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_1, 'b_1')

        buyers_per_ob = 3
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_3.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_3, 'b_3')

        buyers_per_ob = 20
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_20.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_20, 'b_20')

        buyers_per_ob = 30
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_30.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_30, 'b_30')

        buyers_per_ob = 50
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, True, False)
            data_list_50.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_50, 'b_50')


def main_padding():
    global padding
    for cyl2 in range(5):
        data_list_0 = []
        data_list_1 = []
        data_list_2 = []
        padding = 1
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data_list_1.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_1, 'padding-1')

        padding = 2
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_org_price(data_list_1[cyl].si_org, data_list_1[cyl].bj_org)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, False, False)
            data_list_2.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_2, 'padding-2')

        padding = 0
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_org_price(data_list_1[cyl].si_org, data_list_1[cyl].bj_org)
            data.set_g_c(data_list_1[cyl].g_c)
            data.generate_data(False, False, False)
            data_list_0.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_0, 'padding-0')


def main_gc():
    global suppliers_per_rc
    global buyers_per_ob
    global num_suppliers
    global num_buyers
    global g_c_coff
    for cyl2 in range(25):
        data_list_1 = []
        data_list_2 = []
        data_list_05 = []
        data_list_0 = []
        g_c_coff = 0.01
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data_list_1.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_1, 'gc_1')

        g_c_coff = 0.02
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_org_price(data_list_1[cyl].si_org, data_list_1[cyl].bj_org)
            data.set_g_c_org(data_list_1[cyl].g_c_org)
            data.generate_data(False, False, False)
            data_list_2.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_2, 'gc_2')

        g_c_coff = 0.005
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_org_price(data_list_1[cyl].si_org, data_list_1[cyl].bj_org)
            data.set_g_c_org(data_list_1[cyl].g_c_org)
            data.generate_data(False, False, False)
            data_list_05.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_05, 'gc_05')

        g_c_coff = 0
        num_buyers = ob_num * buyers_per_ob
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_1[cyl].rc_loc, data_list_1[cyl].ob_loc,
                              data_list_1[cyl].distance, data_list_1[cyl].rc_sets)
            data.set_org_price(data_list_1[cyl].si_org, data_list_1[cyl].bj_org)
            data.set_g_c_org(data_list_1[cyl].g_c_org)
            data.generate_data(False, False, False)
            data.set_g_c_2(data_list_1[cyl].g_c_2)
            data.gcs = []
            data_list_0.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_0, 'gc_0')


def draw_pic():
    data = GenData()
    data.generate_data(True, True, True)
    writer = pd.ExcelWriter('./instance.xlsx')
    pd.DataFrame(data.rc_loc, columns=['横坐标', '纵坐标']).to_excel(writer, encoding='utf-8', index=True,
                                                                   sheet_name='rc_loc')
    pd.DataFrame(data.ob_loc, columns=['横坐标', '纵坐标']).to_excel(writer, encoding='utf-8', index=True,
                                                                   sheet_name='ob_loc')
    pd.DataFrame(data.distance).to_excel(writer, encoding='utf-8', index=True,
                                         sheet_name='distance')
    pd.DataFrame(data.g_c_org).to_excel(writer, encoding='utf-8', index=True,
                                        sheet_name='g_c_org')
    pd.DataFrame(data.si_org).to_excel(writer, encoding='utf-8', index=True,
                                       sheet_name='si_org')
    pd.DataFrame(data.bj_org).to_excel(writer, encoding='utf-8', index=True,
                                       sheet_name='bj_org')
    pd.DataFrame(data.rc_sets).to_excel(writer, encoding='utf-8', index=True,
                                        sheet_name='rc_sets')
    writer.save()
    fig, ax = plt.subplots()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    x_ = []
    y_ = []
    k_ = []
    l_ = []
    for i in range(rc_num):
        x_.append(data.rc_loc[i][0])
        y_.append(data.rc_loc[i][1])
    for i in range(ob_num):
        k_.append(data.ob_loc[i][0])
        l_.append(data.ob_loc[i][1])
    ax.scatter(x_, y_, marker='^', color='r', label='Residential Communities', s=15)
    ax.scatter(k_, l_, marker='s', color='g', label='Office Buildings', s=15)
    for i in range(0, rc_num):
        ax.text(data.rc_loc[i][0], data.rc_loc[i][1], "RC"+str(i+1))
    for i in range(0, ob_num):
        ax.text(data.ob_loc[i][0], data.ob_loc[i][1], "OB"+str(i+1))
    plt.axis('equal')
    # plt.xlabel('${a}$', font1)
    # plt.ylabel(y_label, font1)
    x = np.linspace(-2, 2, 5000)
    y1 = np.sqrt(2 ** 2 - x ** 2)
    y2 = -np.sqrt(2 ** 2 - x ** 2)
    plt.plot(x, y1, c='k')
    plt.plot(x, y2, c='k')
    plt.legend()
    plt.show()


def main_valuation_var():
    global p_sd
    for cyl2 in range(50):
        data_list_5 = []
        data_list_3 = []
        data_list_10 = []
        p_sd = 5
        for cyl in range(cycle):
            data = GenData()
            data.generate_data(True, True, True)
            data_list_5.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_5, 'psd_5')

        p_sd = 3
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_5[cyl].rc_loc, data_list_5[cyl].ob_loc,
                              data_list_5[cyl].distance, data_list_5[cyl].rc_sets)
            data.set_g_c_org(data_list_5[cyl].g_c_org)
            data.generate_data(False, True, False)
            data_list_3.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_3, 'psd_3')

        p_sd = 10
        for cyl in range(cycle):
            data = GenData()
            data.set_loc_data(data_list_5[cyl].rc_loc, data_list_5[cyl].ob_loc,
                              data_list_5[cyl].distance, data_list_5[cyl].rc_sets)
            data.set_g_c_org(data_list_5[cyl].g_c_org)
            data.generate_data(False, True, False)
            data_list_10.append(data)
        run_exp(num_suppliers, num_buyers, num_parking_zones, data_list_10, 'psd_10')


if __name__ == "__main__":
    main_valuation_var()
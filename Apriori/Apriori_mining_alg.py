import numpy as np


class AprioriMining:
    def __init__(self, observation, total_df_value, abnormal_index, frequent_ratio=0.001, confident_ratio = 0.8, depth=5, rule_bottom_threshold=0.9):

        self.abnormal_index = abnormal_index
        self.map = dict()
        self.re_map = dict()
        self.total_data_set = set()
        self.data_set = set()
        for _ in total_df_value:
            self.total_data_set = self.total_data_set.union(set(_))
        for _ in observation:
            self.data_set = self.data_set.union(set(_))

        for i in enumerate(self.total_data_set):
            self.map[i[1]] = i[0]
            self.re_map[i[0]] = i[1]

        self.search_depth = depth
        self.observation = [[self.map[observation[i][j]] for j in range(0, len(observation[i]))] for i in range(0, len(observation))]
        self.total_df_value = [[self.map[total_df_value[i][j]] for j in range(0, len(total_df_value[i]))] for i in range(0, len(total_df_value))]
        self.data_set = set([self.map[i] for i in self.data_set])

        self.frequent_set = []
        self.rule_bottom_threshold = rule_bottom_threshold
        self.frequent_ratio = frequent_ratio
        self.confident_ratio = confident_ratio
        self.ans_list = []

    def is_frequent_set(self, check_set):
        cnt = 0

        for i in range(0, len(self.observation)):
            if set(check_set).issubset(set(self.observation[i])):
                cnt += 1

        return cnt/len(self.observation) >= self.frequent_ratio

    def is_confident_set(self, check_set):
        cnt = 0
        abnormal_cnt = 0

        for i in range(0, len(self.total_df_value)):
            if set(check_set).issubset(set(self.total_df_value[i])):
                cnt += 1
                if i in self.abnormal_index:
                    abnormal_cnt += 1
        # if cnt == 0:
        #     print("abnormal_node:", [self.re_map[i] for i in check_set])
        # else:
        #     print("check_set:", [self.re_map[i] for i in check_set], "confident_ratio:",  abnormal_cnt/cnt)

        return cnt>0 and abnormal_cnt/cnt >= self.confident_ratio


    def refine(self, frequent_set, data_set):
        next_frequent_set = []
        print("frequent_set:", self.transform(frequent_set))
        for i in frequent_set:
            for j in data_set:
                if j not in i:
                    new_elem = i.copy()
                    new_elem.append(j)
                    next_frequent_set.append(new_elem)

        next_frequent_set = set([tuple(sorted(i)) for i in next_frequent_set])
        next_frequent_set = [list(i) for i in list(next_frequent_set)]
        print("next_frequent_set", self.transform(next_frequent_set))
        return next_frequent_set

    def get_ans(self):
        temp_frequent_set = [[i] for i in self.data_set]

        for i in range(0, self.search_depth):
            temp_frequent_not_confident_set = []
            temp_frequent_and_confident_set = []
            for frequent_set in temp_frequent_set:
                is_frequent_set = self.is_frequent_set(frequent_set)
                is_confident_set = self.is_confident_set(frequent_set)
                if is_frequent_set:
                    if is_confident_set:
                        temp_frequent_and_confident_set.append(frequent_set)
                    else:
                        temp_frequent_not_confident_set.append(frequent_set)

            if len(temp_frequent_and_confident_set) > 0:
                self.ans_list = temp_frequent_and_confident_set
                break

            real_temp_frequent_set = self.transform(temp_frequent_not_confident_set)

            # self.frequent_set.append({i:  real_temp_frequent_set.copy()})
            print(i+1, "element_size = ", len(real_temp_frequent_set),  real_temp_frequent_set)
            temp_frequent_set = self.refine(temp_frequent_not_confident_set, self.data_set)

        return self.transform(self.ans_list)


    def transform(self, mylist):
        return [[self.re_map[j] for j in i]for i in mylist]

    def is_connect_rule(self, number_of_observation_in_sub_data_set, data_set, total_data_set):
        data_set = total_data_set - set(data_set)
        cnt = 0
        for data in self.observation:
            if set(data_set).issubset(set(data)):
                cnt += 1

        return [set(data_set), total_data_set - set(data_set) , number_of_observation_in_sub_data_set/cnt] , number_of_observation_in_sub_data_set/cnt >= self.rule_bottom_threshold

    def get_rules(self, sub_data_set):
        num_of_observation_in_sub_data_set = len([_ for _ in self.observation if sub_data_set.issubset(_)])
        temp_frequent_set = [[i] for i in sub_data_set]
        rules = []
        for i in range(0, len(sub_data_set)-1):
            temp_frequent_set = [_ for _ in temp_frequent_set if self.is_connect_rule
            (num_of_observation_in_sub_data_set, _, sub_data_set)[1]]

            rules.append({i: [self.is_connect_rule(num_of_observation_in_sub_data_set, _, sub_data_set)[0] for _ in temp_frequent_set]})
            temp_frequent_set = self.refine(temp_frequent_set,sub_data_set)
        print(rules)
        return rules


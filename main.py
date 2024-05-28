from math import log2


def calculate_information_for_attribute(attributes, decisions, attribute_index):
    attribute_values = [row[attribute_index] for row in attributes]
    instances_count = len(attribute_values)
    attribute_info = 0.0

    attribute_value_counts = {}
    for value in set(attribute_values):
        attribute_value_counts[value] = attribute_values.count(value)

    for value, count in attribute_value_counts.items():
        value_instances = [i for i, v in enumerate(attribute_values) if v == value]
        value_info = calculate_information_for_decision(decisions, value_instances)
        attribute_info += (count / instances_count) * value_info

    return attribute_info


def calculate_information_for_decision(decisions, value_instances):
    decisions = [decisions[i] for i in value_instances]
    attribute_info = calculate_entropy_attr(decisions)

    return attribute_info


def calculate_information_for_attributes(attributes, decisions):
    attributes_info = {}
    for i in range(len(attributes[0])):
        attribute_info = calculate_information_for_attribute(attributes, decisions, i)
        attributes_info[i] = attribute_info

    return attributes_info


def calculate_entropy_probabilities(probabilities):
    entropy = -sum(p * log2(p) for p in probabilities if p != 0)

    return entropy


def calculate_entropy_attr(attribute):
    probabilities = [attribute.count(value) / len(attribute) for value in set(attribute)]
    entropy = -sum(p * log2(p) for p in probabilities if p != 0)

    return entropy


def calculate_split_info(probabilities):
    split_info = calculate_entropy_probabilities(probabilities)

    return split_info


def calculate_gain_ratio(attribute_values, gain):
    split_info = calculate_entropy_attr(attribute_values)
    gain_ratio = gain / split_info

    return gain_ratio


def main():
    data1 = [
        [0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [2, 2, 0, 1, 1],
        [2, 2, 1, 0, 2],
        [2, 2, 0, 0, 1],
        [1, 2, 0, 1, 0],
        [2, 1, 1, 1, 2],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1]]

    data2 = [
        ['old', 'yes', 'swr', 'down'],
        ['old', 'no', 'swr', 'down'],
        ['old', 'no', 'hwr', 'down'],
        ['mid', 'yes', 'swr', 'down'],
        ['mid', 'yes', 'hwr', 'down'],
        ['mid', 'no', 'hwr', 'up'],
        ['mid', 'no', 'swr', 'up'],
        ['new', 'yes', 'swr', 'up'],
        ['new', 'no', 'hwr', 'up'],
        ['new', 'no', 'swr', 'up']]

    data3 = [
        [1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1]]

    data4 = []

    with open('car.data', 'r') as file:
        for line in file:
            line_data = line.strip().split(',')
            data4.append(line_data)

    dataset = data3

    conditional_attributes = [row[:-1] for row in dataset]
    decisions = [row[-1] for row in dataset]

    entropy = calculate_entropy_attr(decisions)

    print(f"Entropia zbioru: {entropy}")

    attributes_info = calculate_information_for_attributes(conditional_attributes, decisions)

    for attribute, info in attributes_info.items():
        gain = entropy - info
        gain_ratio = calculate_gain_ratio([row[attribute] for row in conditional_attributes], gain)
        print(f"Info(A{attribute+1},T): {info}")
        print(f"Gain(A{attribute+1},T): {gain}")
        print(f"GainRatio(A{attribute+1},T): {gain_ratio}")


if __name__ == "__main__":
    main()

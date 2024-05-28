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
    if split_info == 0:
        gain_ratio = 0
    else:
        gain_ratio = gain / split_info

    return gain_ratio


class DecisionNode:
    def __init__(self, attribute=None, decision=None, value=None):
        self.attribute = attribute
        self.decision = decision
        self.value = value
        self.children = {}


def stop_condition(gain_ratios):
    return all(gain_ratio == 0 for gain_ratio in gain_ratios) or gain_ratios == []


def choose_best_attr(gain_ratios, dataset):
    best_attr_index = gain_ratios.index(max(gain_ratios))
    return [row[best_attr_index] for row in dataset], best_attr_index


def calculate_gain_ratios(dataset):
    gain_ratios = []
    if len(dataset) == 0:
        return gain_ratios
    conditional_attributes = [row[:-1] for row in dataset]
    decisions = [row[-1] for row in dataset]
    entropy = calculate_entropy_attr(decisions)
    attributes_info = calculate_information_for_attributes(conditional_attributes, decisions)
    for attribute, info in attributes_info.items():
        gain = entropy - info
        gain_ratios.append(calculate_gain_ratio([row[attribute] for row in conditional_attributes], gain))
    return gain_ratios


def split_attr(dataset, best_attr_index, value):
    return [row[:-1] + [row[-1]] for row in dataset if row[best_attr_index] == value]


def decision_class(data):
    return data[-1][-1]


def construct_tree(dataset, node=None, value=None):
    if node is None:
        node = DecisionNode(value=value)

    gain_ratios = calculate_gain_ratios(dataset)

    if not stop_condition(gain_ratios):
        best_attr, best_attr_index = choose_best_attr(gain_ratios, dataset)
        node.attribute = best_attr_index
        for i, val in enumerate(set(best_attr)):
            child_dataset = split_attr(dataset, best_attr_index, val)
            node.children[i] = construct_tree(child_dataset, value=val)
    else:
        node.decision = decision_class(dataset)
        node.value = value

    return node


def visualize_tree_to_file(node, file, indent=0):
    if node.decision is not None:
        file.write(" D: " + str(node.decision) + "\n")
    else:
        file.write(f"Atrybut: {node.attribute + 1}\n")
        for value, child_node in node.children.items():
            file.write(" " * indent + f"{' ' * (indent + 10)}{child_node.value} -> ")
            visualize_tree_to_file(child_node, file, indent + 6)


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

    dataset = data4
    tree = construct_tree(dataset)

    with open('decision_tree_visualization.txt', 'w') as f:
        visualize_tree_to_file(tree, file=f)


if __name__ == "__main__":
    main()

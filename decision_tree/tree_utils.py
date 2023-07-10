from tree import Tree

def plot_tree(tree: Tree):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    node_positions = calculate_node_positions(tree.root, x=0, y=0)
    plot_node(ax, tree.root, node_positions)
    ax.axis('off')
    return fig, ax

def plot_node(ax, node, node_positions):
    if node is None:
        return

    position = node_positions[node]

    # Draw the node box
    if node.is_leaf:
        ax.text(position[0], position[1], f"Impurity: {node.impurity:.3f} \n samples: {node.n_samples}\n LEAF WITH VAL: {node.value:.3f}",
            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
    else:
        ax.text(position[0], position[1], f"Decision WITH x{node.split_idx} <= {node.threshold:.3f}\n Impurity: {node.impurity:.3f} \n samples: {node.n_samples}",
            ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))

    # Draw edges and child nodes recursively
    if node.left_child is not None:
        ax.plot([position[0], node_positions[node.left_child][0]], [position[1], node_positions[node.left_child][1]], color='black')
        plot_node(ax, node.left_child, node_positions)
    if node.right_child is not None:
        ax.plot([position[0], node_positions[node.right_child][0]], [position[1], node_positions[node.right_child][1]], color='black')
        plot_node(ax, node.right_child, node_positions)


def calculate_node_positions(node, x, y):
    if node is None:
        return {}

    dx = 1
    dy = 1

    left_positions = calculate_node_positions(node.left_child, 2 * x - dx , y - dy)
    right_positions = calculate_node_positions(node.right_child, 2 * x + dx, y - dy)

    position = (x, y)

    node_positions = {**left_positions, **right_positions, node: position}

    return node_positions

def print_tree(tree: Tree):
    queue = []
    queue.append(tree.root)
    while len(queue) > 0:
        node = queue.pop()
        if node:
            print(f"Depth: {node.depth}")
            print(f"Impurity: {node.impurity}")
            print(f"samples: {node.n_samples}")
            if node.is_leaf:
                print(f"LEAF WITH VAL: {node.value}")
            else:
                print(f"Decision WITH x{node.split_idx} <= {node.threshold}")
            print("") # spacing
            queue.append(node.left_child)
            queue.append(node.right_child)
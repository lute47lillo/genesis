from graphviz import Digraph

# Define the TreeNode class
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Function to manually construct the specific tree
def construct_specific_tree():
    # Level 0 (Root)
    root = TreeNode('+')

    # Level 1
    root.left = TreeNode('*')
    root.right = TreeNode('log')

    # Level 2
    root.left.left = TreeNode('1')
    root.left.right = TreeNode('x')
    root.right.left = TreeNode('1')
    # root.right.right = TreeNode('x')

    return root

# Function to add nodes and edges to the graph
def add_nodes_edges(tree, dot, parent_id=None):
    if tree is not None:
        # Create a unique identifier for each node
        node_id = str(id(tree))
        # Add the current node to the graph
        dot.node(node_id, tree.value, shape='circle', style='filled', fillcolor='lightblue')
        # If there's a parent, create an edge from parent to current node
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        # Recursively add child nodes
        add_nodes_edges(tree.left, dot, node_id)
        add_nodes_edges(tree.right, dot, node_id)

# Function to visualize the tree
def visualize_tree(tree, filename='specific_gp_tree'):
    dot = Digraph(comment='Specific GP-based Tree')
    dot.attr(rankdir='TB')  # Top to Bottom layout
    add_nodes_edges(tree, dot)
    # Render the tree to a file
    dot.format = 'png'  # Options: 'png', 'pdf', 'svg', etc.
    dot.render(filename, view=False)
    print(f"Specific GP tree image saved as {filename}.png")

if __name__ == "__main__":
    # Construct the specific GP tree
    specific_gp_tree = construct_specific_tree()

    # Visualize and save the tree image
    visualize_tree(specific_gp_tree, filename='specific_gp_tree_depth2')

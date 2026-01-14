
import bpy

scene = bpy.context.scene
scene.use_nodes = True
tree = bpy.data.node_groups.new(type="CompositorNodeTree", name="Compositing Nodetree")
scene.compositing_node_group = tree

print("Creating NodeGroupOutput...")
output_node = tree.nodes.new(type="NodeGroupOutput")

print(f"Initial inputs: {[i.name for i in output_node.inputs]}")

print("Adding 'Image' input to tree interface...")
if hasattr(tree, "interface"):
    # Blender 4.0+ / 5.0 interface API
    socket = tree.interface.new_socket(name="Image", in_out='INPUT', socket_type='NodeSocketColor')
    print(f"Added interface socket: {socket.name}")
elif hasattr(tree, "inputs"):
    # Older API
    tree.inputs.new("NodeSocketColor", "Image")

print(f"Updated inputs: {[i.name for i in output_node.inputs]}")

print("Adding 'Alpha' input to tree interface...")
if hasattr(tree, "interface"):
    socket = tree.interface.new_socket(name="Alpha", in_out='INPUT', socket_type='NodeSocketFloat')
    print(f"Added interface socket: {socket.name}")
else:
    tree.inputs.new("NodeSocketFloat", "Alpha")

print(f"Final inputs: {[i.name for i in output_node.inputs]}")

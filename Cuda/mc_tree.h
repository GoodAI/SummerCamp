#ifndef MC_TREE_H
#define MC_TREE_H


typedef enum { Chance, Decision } node_type_enum_type;

//struct mc_key_val;



struct mc_key_val_s;

typedef struct {
	int type;
	float exploration_constant;
	float unexplored_bias;
	float mean = 0.0;
	int visits = 0;
	
	struct mc_key_val_s* children;//pointer to array with pointers to children to childs.
	int children_array_size;
	int children_count;

} mc_tree_node;

struct mc_key_val_s {
	int key;
	mc_tree_node* val;
};

typedef struct mc_key_val_s mc_key_val;


__device__ mc_tree_node* create_mc_node(int type);

__device__ mc_tree_node* get_child(mc_tree_node* node, int key);

__device__ int select_action(mc_tree_node * me);

__device__ float sample(int horizon, mc_tree_node* me);

#endif
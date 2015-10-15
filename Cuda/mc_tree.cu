
#include "mc_tree.h"

#include "ctw_context_tree.h"
#include "cuda_runtime.h"
#include <math.h>
#include "vars.h"

__device__ mc_tree_node* create_mc_node(int type){
	mc_tree_node* node = (mc_tree_node*)malloc(sizeof(mc_tree_node));
	node->visits = 0;
	node->type = type;
	node->exploration_constant = 2.0;
	node->unexplored_bias = 100000.0;
	node->mean = 0.0;

	node->children_array_size = 1;//TODO: put something larger here (compute it from number of actions/observations?)
	node->children_count = 0;
	node->children = (mc_key_val*)malloc(sizeof(mc_key_val)*node->children_array_size);
	return node;
}

__device__ mc_tree_node* get_child(mc_tree_node* node, int key){
	//todo:rename to mc_get_child
	mc_key_val* children = node->children;
	for (int i = 0; i < node->children_count; i++){
		if (children[i].key == key){
			return children[i].val;
		}
	}
	return NULL;
}

__device__ mc_tree_node* mc_tree_get_or_create_child(mc_tree_node* node, int key, int type){
	mc_tree_node* child = get_child(node, key);
	if (child == NULL){
		child = create_mc_node(type);

		if (node->children_array_size <= node->children_count){//resize
			int old_size = node->children_array_size;
			node->children_array_size *= 2;
			mc_key_val *new_array = (mc_key_val*)malloc(sizeof(mc_key_val)*node->children_array_size);
			
			memcpy(new_array, node->children, sizeof(mc_key_val)*old_size);
			free(node->children);
			node->children = new_array;
		}

		mc_key_val item;
		item.key = key;
		item.val = child;

		node->children[node->children_count] = item;
		node->children_count++;
	}
	return child;
}

__device__ int select_action(mc_tree_node * me){
	
	float explore_bias= (float)AGENT->horizon * AGENT->maximum_action;

	float exploration_numerator = me->exploration_constant * logf(me->visits);
	int best_action = -1;
	float best_priority = -INFINITY;

	for (int action = 0; action <= AGENT->maximum_action; action++){
		mc_tree_node * node = get_child(me, action);//refact: n^2 search here
		float priority;

		if (node == NULL || node->visits == 0){
			priority = me->unexplored_bias;
		}
		else{
			priority = node->mean + explore_bias * sqrtf(exploration_numerator / node->visits);
		}

		if (priority > best_priority){ // todo: randomize (+randomDouble(0,0.001))
			best_action = action;
			best_priority = priority;
		}

	}
	return best_action;

}


__device__ float sample(int horizon, mc_tree_node* me) {

	float reward = 0.0;

	if (horizon == 0) {
		return (int)reward;
	}
	else if (me->type == Chance) {
		
		int percept = generate_percept_and_update();

		int observation = percept & (( 1 << ( AGENT->observation_bits + 1 )) - 1);
		int randomReward = percept>>AGENT->observation_bits;

		mc_tree_node* obs_child = mc_tree_get_or_create_child(me, observation, Decision);

		
		reward = randomReward + sample(horizon - 1, obs_child);
	}
	else if (me->visits == 0) //unvisited decision node or we have exceeded maximum tree depth
	{
		reward = playout(horizon);
	}
	else { //Previously visited decision node
		int action = select_action(me);
		model_update_action(action);

		mc_tree_node* action_child = mc_tree_get_or_create_child(me, action, Chance);

		reward = sample(horizon, action_child);   //it is not clear if not horizon-1. (asks pyaixi)
	}

	float visits_f = me->visits;
	me->mean = (reward + (visits_f*me->mean)) / (1.0 + visits_f);
	me->visits++;

	return reward;
}
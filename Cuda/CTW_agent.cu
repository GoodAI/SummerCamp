
#include "ctw_context_tree.h"
#include "cuda_runtime.h"
#include "vars.h"
#include "CTW_agent.h"
#include "utils.h"
#include "mc_tree.h"
//#include <stdint.h>

__device__ ctw_agent* create_agent(
	int depth,
	int mc_simulations,
	int horizon,
	int maximum_action,//maximum_X should be normalized range 0-...
	int maximum_reward,
	int maximum_observation
	){
	//note: learning period is not implemented here (have to be done on level of interaction loop)

	ctw_agent*agent = (ctw_agent*)malloc(sizeof(ctw_agent));
	AGENT = agent;

	agent->last_update = ActionUpdate;
	agent->depth = depth;
	agent->mc_simulations = mc_simulations;

	agent->context_tree = create_ct_tree(depth);
	
	TREE = agent->context_tree;//n: this may not be needed (since it is already done in create_ct_tree)

	agent->horizon = horizon;

	agent->total_reward = 0;

	agent->age = 0;

	agent->maximum_action = maximum_action;
	agent->maximum_reward = maximum_reward;
	agent->maximum_observation = maximum_observation;

	agent->action_bits = BitsNeeded(agent->maximum_action);
	agent->reward_bits = BitsNeeded(agent->maximum_reward);
	agent->observation_bits = BitsNeeded(agent->maximum_observation);

	agent->percept_bits = agent->reward_bits + agent->observation_bits;
	return agent;
}

__device__ double average_reward() {
	//return reward that is normalized to 0-maximum_reward. To unnormalize you must add minimum reward
	if (AGENT->age > 0)
	{
		return AGENT->total_reward/ AGENT->age;
	}
	else {
		return 0.0;
	}
}

__device__ int encode_percept(int observation, int reward)
{
	return observation | (reward << AGENT->observation_bits);
}

__device__ void model_update_percept(int observation, int reward)
{
	assert(AGENT->last_update == ActionUpdate);

	int perceptSymbols = encode_percept(observation, reward);

	if (TREE->history_count >= TREE->depth){
		update_tree(perceptSymbols, AGENT->percept_bits);
	}
	else{
		update_tree_history_symbols(perceptSymbols, AGENT->percept_bits);
	}


	AGENT->total_reward += reward;
	AGENT->last_update = PerceptUpdate;
}


__device__ void model_update_action(int action)
{
	assert(action >= 0 && action <= AGENT->maximum_action);
	assert(AGENT->last_update == PerceptUpdate);

	update_tree_history_symbols(action, AGENT->action_bits);

	AGENT->age++;
	AGENT->last_update = ActionUpdate;
}


__device__ int generate_percept_and_update() {
	int percept = generate_random_symbols_and_update(AGENT->percept_bits);
	AGENT->last_update = PerceptUpdate;
	int reward = percept >> AGENT->observation_bits;
	AGENT->total_reward += reward;
	return percept;

}


__device__ void model_revert(ct_tree_undo* undo_instance)
{
	while (TREE->history_count > undo_instance->history_count){
		if (AGENT->last_update == PerceptUpdate)
		{
			revert_tree(AGENT->percept_bits);
			AGENT->last_update = ActionUpdate;
		}
		else {
			revert_tree_history(AGENT->action_bits);
			AGENT->last_update = PerceptUpdate;
		}
	}

	AGENT->age = undo_instance->age;
	AGENT->total_reward = undo_instance->total_reward;
	AGENT->last_update = undo_instance->last_update;//is this step needed? we have already set last_update above
}

__device__ int generate_random_action(){
	return rand_int_range(0,AGENT->maximum_action);
}

__device__ float playout(int horizon) {
	//idea: run this several times (in several threads?) and then avg results
	float total_reward = 0.0;

	for (int i = 0; i < horizon; i++) {
		int action = generate_random_action();//later: improved playout policy?
		model_update_action(action);
		int percept = generate_percept_and_update();
		int reward = percept >> AGENT->observation_bits;
		total_reward += reward;
	}
	return total_reward;
}


__device__ int search() {
	ct_tree_undo * undo_instance = backup_tree();
	mc_tree_node* mc_root = create_mc_node(Decision);

	for (int i = 0; i < AGENT->mc_simulations; i++) {
		sample(AGENT->horizon, mc_root);
		model_revert(undo_instance);
	}

	int best_action = -1;
	float best_mean = -INFINITY;
	for (int i = 0; i < mc_root->children_count; i++){
		mc_key_val* item = &(mc_root->children[i]);
		int action = item->key;
		mc_tree_node * child = item->val;
		float mean = child->mean+rand_range(0, 0.0001);
		if (mean > best_mean){
			best_mean = mean;
			best_action = action;
		}
	}
	return best_action;
}
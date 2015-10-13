#ifndef CTW_AGENT_H
#define CTW_AGENT_H
#include "ctw_context_tree.h"

typedef enum { ActionUpdate, PerceptUpdate } update_enum_type;
//__device__ update_enum_type ActionUpdate = ActionUpdate;//are these two needed?
//__device__ update_enum_type PerceptUpdate = PerceptUpdate;

typedef struct {
	int depth;
	int mc_simulations;
	ct_tree* context_tree;

	int age;

	int horizon;

	//todo: options

	int last_update;
	//for now we are ignoring learning period - it is not needed at the moment
//	int learning_period;
	
	float total_reward;

	int maximum_action;// these values are already normalized, 
	int maximum_reward;
	int maximum_observation;

	int action_bits;
	int reward_bits;
	int observation_bits;
	
	int percept_bits;

} ctw_agent;


__device__ ctw_agent* create_agent(
	int depth,
	int mc_simulations,
	int horizon,
	int maximum_action,
	int maximum_reward,
	int maximum_observation
	);

__device__ double average_reward();


__device__ int GenerateRandomObservation();
__device__ int GenerateRandomAction();
__device__ int GenerateRandomReward();

__device__ void model_update_percept(int observation, int reward);

__device__ int GeneratePercept();

__device__ int GeneratePerceptAndUpdate();

__device__ int Search();





#endif

#include "ctw_context_tree.h"
#include "cuda_runtime.h"
#include "vars.h"
#include "CTW_agent.h"
#include "utils.h"
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
	
	TREE = agent->context_tree;//n: this may not be needed (since it is already done in create_ct_tree

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
	if (AGENT->age > 0)
	{
		return AGENT->total_reward/ AGENT->age;
	}
	else {
		return 0.0;
	}
}


__device__ int GenerateRandomObservation() {
	//TODO
}
__device__ int GenerateRandomAction()
{
	//TODO
}
__device__ int GenerateRandomReward()
{
	//TODO
}

__device__ int encode_percept(int observation, int reward)
{
	return reward | (observation << AGENT->reward_bits);
}

__device__ void model_update_percept(int observation, int reward)
{
	assert(AGENT->last_update == ActionUpdate);

	int perceptSymbols = encode_percept(observation, reward);

	update_tree(perceptSymbols, AGENT->percept_bits);

	AGENT->total_reward += reward;
	AGENT->last_update = PerceptUpdate;
}

__device__ int GeneratePercept() { //used to return tuple<int, int >
	//TODO
	/*
	int observation = Utils.RandomElement(this.Environment.ValidObservations);
	int reward = Utils.RandomElement(this.Environment.ValidRewards);
	return new Tuple<int, int>(observation, reward);
	*/
}

__device__ int GeneratePerceptAndUpdate() { //used to return tuple<int, int >
/*	int[] perceptSymbols = this.ContextTree.GenerateRandomSymbolsAndUpdate(this.Environment.perceptBits());

	Tuple<int, int> OandR = this.decode_percept(perceptSymbols);


	int observation = OandR.Item2;
	int reward = OandR.Item1;

	this.TotalReward += reward;
	this.LastUpdate = PerceptUpdate;
	
	return new Tuple<int, int>(observation, reward);*/
}

__device__ int Search() {
/*	CtwContextTreeUndo undoInstance = new CtwContextTreeUndo(this);
	MonteCarloSearchNode searchTree = new MonteCarloSearchNode(MonteCarloSearchNode.DecisionNode);
	for (int i = 0; i < this.McSimulations; i++) {
		searchTree.Sample(this, this.Horizon, 0);
		this.model_revert(undoInstance);
	}

	searchTree.PrintBs();


	int bestAction = -1;
	double bestMean = double.NegativeInfinity;
	foreach(int action in this.Environment.ValidActions) {

		if (!searchTree.Children.ContainsKey(action)) {
			continue;
		}

		double mean = searchTree.Children[action].Mean + Utils.RandomDouble(0, 0.0001);
		if (mean > bestMean) {
			bestMean = mean;
			bestAction = action;
		}
	}
	return bestAction;*/
}


__device__ void model_revert(ct_tree_undo* undo_instance)
{
	while (TREE->history_count>undo_instance->history_count){
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
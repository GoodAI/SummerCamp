using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using LSMModule.LSM.Tasks;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace LSMModule {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Alpha release</status>
    /// <summary>Liquid State Machine node</summary>
    /// <description>
    /// Liquid State Machine node - the key node of this module<br></br>
    /// - tranfers binary input into an output, which also takes into account last few elements of the sequence<br></br>
    /// - edges of the network are generated randomly with constrains put by different topologies<br></br>
    /// - neurons are of type integrate-and-fire(IF)<br></br>
    /// - recommended connectivity of LSM is said to be between 5-10% based on used topology<br></br>
    /// - LSM can be either spiking or non-spiking<br></br>
    /// - inner state and output of neurons are computed used equations discribed in LSMOutputTask<br></br>
    /// - this implementation also allows you to make LSM spike internally more than once in one step<br></br>
    /// - as output of this LSM we take current state of all the neurons in current step
    /// </description>
    class LiquidStateMachine : MyWorkingNode {

        [YAXSerializableField(DefaultValue = 0.1f)]
        [MyBrowsable, Category("\tNetwork")]
        public virtual float Connectivity { get; set; }

        [YAXSerializableField(DefaultValue = 144)]
        [MyBrowsable, Category("\tNetwork")]
        public virtual int Inputs { get; set; }

        [YAXSerializableField(DefaultValue = 0.5f)]
        [MyBrowsable, Category("\tNetwork")]
        public virtual float Threshold { get; set; }

        [YAXSerializableField(DefaultValue = 10)]
        [MyBrowsable, Category("\tNetwork")]
        public virtual int InnerCycle { get; set; }

        [YAXSerializableField(DefaultValue = -65)]
        [MyBrowsable, Category("\tNeurons")]
        public virtual int InitState { get; set; }

        [YAXSerializableField(DefaultValue = -130)]
        [MyBrowsable, Category("\tNeurons")]
        public virtual int RefractoryState { get; set; }

        [YAXSerializableField(DefaultValue = 1.3f)]
        [MyBrowsable, Category("\tNeurons")]
        public virtual float Refractory { get; set; }

        [YAXSerializableField(DefaultValue = true)]
        [MyBrowsable, Category("\tSpike")]
        public virtual bool Spikes { get; set; }

        [YAXSerializableField(DefaultValue = 80)]
        [MyBrowsable, Category("\tSpike")]
        public virtual int SpikeSize { get; set; }

        [YAXSerializableField(DefaultValue = 28)]
        [MyBrowsable, Category("\tPattern")]
        public virtual int PatternLength { get; set; }

        [YAXSerializableField(DefaultValue = 20)]
        [MyBrowsable, Category("Misc")]
        public int OutputColumnHint { get; set; }

        public enum NeuronTypeEnum {
            IF,
            IF2
        }

        [YAXSerializableField(DefaultValue = NeuronTypeEnum.IF)]
        [MyBrowsable, Category("\tLayer")]
        public virtual NeuronTypeEnum NeuronType { get; set; }

        #region Memory blocks
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyPersistable]
        public MyMemoryBlock<float> Weights { get; set; } //done
        public MyMemoryBlock<float> EdgeInputs { get; set; }
        public MyMemoryBlock<float> ImageInput { get; set; }
        [MyPersistable]
        public MyMemoryBlock<int> ImageOutput { get; set; }
        public MyMemoryBlock<float> InnerStates { get; set; }
        public MyMemoryBlock<float> NeuronOutputs { get; set; }
        [MyPersistable]
        public MyMemoryBlock<int> OutputsIndex { get; set; }

        #endregion

        [MyTaskGroup("Init")]
        public LSMRandomInitTask RandomInitTask { get; private set; }
        [MyTaskGroup("Init")]
        public LSMMaassInitTask MaassInitTask { get; private set; }
        public LSMComputeTask ComputeTask { get; private set; }
        public LSMOutputTask OutputTask { get; private set; }

        public int Neurons;

        public override void UpdateMemoryBlocks() {

            // Calculates number of neurons based on used topology
            // Not sure whether this approach is correct, but it works

            if (RandomInitTask != null && RandomInitTask.Enabled) {
                Neurons = RandomInitTask.getNeurons();
            } else if (MaassInitTask != null && MaassInitTask.Enabled) {
                Neurons = MaassInitTask.getNeurons();
            } else {
                Neurons = 0;
            }

            Output.Count = Neurons - Inputs;
            Output.ColumnHint = OutputColumnHint;
            OutputsIndex.Count = Neurons - Inputs;

            Weights.Count = Neurons*Neurons;
            Weights.ColumnHint = Neurons;

            EdgeInputs.Count = Neurons*Neurons;
            Weights.ColumnHint = Neurons;

            ImageInput.Count = Neurons;
            ImageOutput.Count = Inputs;
            ImageInput.ColumnHint = OutputColumnHint;
            ImageOutput.ColumnHint = 12;

            InnerStates.Count = Neurons;
            NeuronOutputs.Count = Neurons;
            InnerStates.ColumnHint = OutputColumnHint;
            NeuronOutputs.ColumnHint = OutputColumnHint;

        }

    }
}

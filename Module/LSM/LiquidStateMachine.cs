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
    /// <meta>mv</meta>
    /// <status>Work in progress</status>
    /// <summary>Liquid State Machine node</summary>
    /// <description>TBA</description>
    class LiquidStateMachine : MyWorkingNode {

        public const float SPIKE_SIZE = 1;
        public const int INNER_CYCLE = 1;

        [YAXSerializableField(DefaultValue = 0.1f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float Connectivity { get; set; }

        [YAXSerializableField(DefaultValue = 144)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int Inputs { get; set; }

        [YAXSerializableField(DefaultValue = 0.5f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float Threshhold { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float A { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float B { get; set; }

        [YAXSerializableField(DefaultValue = true)]
        [MyBrowsable, Category("\tLayer")]
        public virtual bool Spikes { get; set; }

        [YAXSerializableField(DefaultValue = 20)]
        [MyBrowsable, Category("Misc")]
        public int OutputColumnHint { get; set; }

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

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> FileHeader {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        public MyMemoryBlock<float> Weights { get; set; } //done
        public MyMemoryBlock<float> EdgeInputs { get; set; }
        public MyMemoryBlock<float> ImageInput { get; set; }
        public MyMemoryBlock<int> ImageOutput { get; set; }
        public MyMemoryBlock<float> InnerStates { get; set; }
        public MyMemoryBlock<float> NeuronOutputs { get; set; }
        public MyMemoryBlock<float> ImageSpikeProbabilities { get; set; }
        public MyMemoryBlock<int> IsInput { get; set; }

        #endregion

        [MyTaskGroup("Init")]
        public LSMRandomInitTask RandomInitTask { get; private set; }
        [MyTaskGroup("Init")]
        public LSMMaassInitTask MaassInitTask { get; private set; }
        public LSMComputeTask ComputeTask { get; private set; }
        public LSMOutputTask OutputTask { get; private set; }

        public int Neurons;

        public override void UpdateMemoryBlocks() {
            if (RandomInitTask != null && RandomInitTask.Enabled) {
                MyLog.DEBUG.WriteLine("random");
                Neurons = RandomInitTask.getNeurons();
            } else if (MaassInitTask != null && MaassInitTask.Enabled) {
                MyLog.DEBUG.WriteLine("maass");
                Neurons = MaassInitTask.getNeurons();
            } else {
                MyLog.DEBUG.WriteLine("wtf");
                Neurons = 0;
            }

            Output.Count = Neurons-Inputs;
            Output.ColumnHint = OutputColumnHint;

            FileHeader.Count = 7;

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

            ImageSpikeProbabilities.Count = Inputs;
            ImageSpikeProbabilities.ColumnHint = 24;

            IsInput.Count = Neurons;

        }

        //public override void Validate(MyValidator validator) {
        //    base.Validate(validator);
        //    validator.AssertError(Neurons > 0, this, "Number of neurons should be > 0");
        //    //validator.AssertWarning(Connection != ConnectionType.NOT_SET, this, "ConnectionType not set for " + this);
        //}

    }
}

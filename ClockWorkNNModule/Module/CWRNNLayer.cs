using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;
using GoodAI.Core.Signals;
using System.Drawing;
using YAXLib;
using ManagedCuda;
using System.Diagnostics;
using GoodAI.Modules.Transforms;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Core;
using CWRNN.Tasks;

namespace CWRNN
{
    public enum PeriodEnum
    {
        ALL_SAME,
        EXPONENTIAL,
        FIBONACCI,
        LOGARITHMIC,
        RANDOM_SERIE,
        QUADRATIC
    };
    /// <author>Alena Moravova</author>
    /// <meta>am</meta>
    /// <status>Working</status>
    /// <summary>Clockwork recurrent network trained by Real-Time Recurrent Learning</summary>
    /// <description>CWRNN network with partially connected recurrent hidden layer trained by Real-Time Recurrent Learning (RTRL) algorithm. <br />
    ///              Parameters:
    ///              <ul>
    ///                 <li>INPUT_UNITS: Read-only number of units in input layer</li>
    ///                 <li>NeuronGroups: Number of neuron groups in hidden layer</li>
    ///                 <li>NeuronsPerGroup: Number of units in one neuron group</li>
    ///                 <li>HIDDEN_UNITS: Read-only Number of units in hidden layer</li>
    ///                 <li>OUTPUT_UNITS: Read-only number of units in output layer</li>
    ///              </ul>
    ///              
    ///              I/O:
    ///              <ul>
    ///                 <li>Input: Input vector copied to activation of input layer units and propagated through the network</li>
    ///                 <li>Target: Desired activation of output layer units </li>
    ///                 <li>Output: Activation of output layer units </li>
    ///              </ul>
    ///              
    /// </description>
    [YAXSerializeAs("Clockwork RecurrentNetwork")]
    public class CWRNNLayer : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Target { get { return GetInput(1); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [YAXSerializableField(DefaultValue = 4)]
        [MyBrowsable, Category("\tLayer")]
        public int NeuronGroups { get; set; }

        [YAXSerializableField(DefaultValue = 6)]
        [MyBrowsable, Category("\tLayer")]
        public int NeuronsPerGroup { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("Settings")]
        public int contextByActivations { get; set; }


        [YAXSerializableField(DefaultValue = PeriodEnum.EXPONENTIAL)]
        [MyBrowsable, Category("Structure")]
        public PeriodEnum Period { get; set; }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("Structure")]
        public ActivationFunctionType ACTIVATION_FUNCTION { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int INPUT_UNITS { get; protected set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int HIDDEN_UNITS { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int OUTPUT_UNITS { get; protected set; }

        [MyBrowsable, Category("Observers")]
        [YAXSerializableField(DefaultValue = 5), YAXElementFor("Observers")]
        public int ColumnHint { get; set; }

        [MyPersistable]
        public MyMemoryBlock<float> InputWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> InputWeightDeltas { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> RecurrentWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> RecurrentWeightDeltas { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> OutputWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> OutputWeightDeltas { get; protected set; }

        public MyMemoryBlock<float> InputWeightRTRLDerivatives { get; protected set; }
        public MyMemoryBlock<float> PreviousInputWeightRTRLDerivatives { get; protected set; }

        public MyMemoryBlock<float> RecurrentWeightRTRLDerivatives { get; protected set; }
        public MyMemoryBlock<float> PreviousRecurrentWeightRTRLDerivatives { get; protected set; }

        public MyMemoryBlock<float> HiddenActivations { get; protected set; }
        public MyMemoryBlock<float> OutputActivations { get; protected set; }
        public MyMemoryBlock<float> ContextActivations { get; protected set; }

        public MyMemoryBlock<float> PreviousHiddenActivations { get; protected set; }

        public MyMemoryBlock<float> HiddenActivationDerivatives { get; protected set; }
        public MyMemoryBlock<float> OutputActivationDerivatives { get; protected set; }

        public MyMemoryBlock<float> OutputDeltas { get; protected set; }

        public MyMemoryBlock<int> Periods { get; set; }
        public MyMemoryBlock<int> ActiveGroups { get; set; }

        //TASKS
        public CWInitLayerTask InitNetwork { get; protected set; }
        public CWFeedForwardTask Feedforward { get; protected set; }
        public SetContextTask SetContext { get; protected set; }
        public CWRTRLTask RTRL { get; protected set; }


        public override void UpdateMemoryBlocks()
        {
            if (Input != null && Target != null)
            {
                INPUT_UNITS = Input.Count;
                OUTPUT_UNITS = Target.Count;
                Output.Count = OUTPUT_UNITS;
                Output.ColumnHint = ColumnHint;

                HIDDEN_UNITS = NeuronGroups * NeuronsPerGroup;

                HiddenActivations.Count = HIDDEN_UNITS;
                ContextActivations.Count = HIDDEN_UNITS * NeuronGroups;

                HiddenActivationDerivatives.Count = HIDDEN_UNITS;
                PreviousHiddenActivations.Count = HIDDEN_UNITS;
                OutputActivations.Count = OUTPUT_UNITS;
                OutputActivationDerivatives.Count = OUTPUT_UNITS;
                OutputDeltas.Count = OUTPUT_UNITS;

                InputWeights.Count = HIDDEN_UNITS * INPUT_UNITS;
                InputWeightDeltas.Count = HIDDEN_UNITS * INPUT_UNITS;

                RecurrentWeights.Count = HIDDEN_UNITS * HIDDEN_UNITS;
                RecurrentWeightDeltas.Count = HIDDEN_UNITS * HIDDEN_UNITS;

                OutputWeights.Count = OUTPUT_UNITS * HIDDEN_UNITS;
                OutputWeightDeltas.Count = OUTPUT_UNITS * HIDDEN_UNITS;

                InputWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * INPUT_UNITS;
                PreviousInputWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * INPUT_UNITS;

                RecurrentWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * HIDDEN_UNITS;
                PreviousRecurrentWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * HIDDEN_UNITS;

                Periods.Count = NeuronGroups;
                ActiveGroups.Count = NeuronGroups;

                // make an even number of weights for the cuda random initialisation
                if (InputWeights.Count % 2 != 0)
                    InputWeights.Count++;
                if (RecurrentWeights.Count % 2 != 0)
                    RecurrentWeights.Count++;
                if (OutputWeights.Count % 2 != 0)
                    OutputWeights.Count++;
            }
        }
    }
}

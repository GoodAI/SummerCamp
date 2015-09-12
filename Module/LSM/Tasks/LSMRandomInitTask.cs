using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace LSMModule.LSM.Tasks {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Work in progress</status>
    /// <summary>Task for initialization of network with Random topology</summary>
    /// <description>
    /// Starting initialization for LSM with Random topology:<br></br>
    /// - inputs are selected randomly<br></br>
    /// - each node has c% of all output/input edges, randomly selected with random weights, where c is connectivity of the graph<br></br>
    ///   - whether it is output or input edges can be changed in Brain Simulator
    /// </description>
    [Description("Init random network"), MyTaskInfo(OneShot = true)]
    class LSMRandomInitTask : MyTask<LiquidStateMachine> {

        public enum IOEnum {
            input,
            output
        }

        [YAXSerializableField(DefaultValue = IOEnum.output)]
        [MyBrowsable, Category("\tLayer")]
        public virtual IOEnum ConnectivityType { get; set; }

        [YAXSerializableField(DefaultValue = 400)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int Neurons { get; set; }

        public override void Init(int nGPU) {
        }

        public override void Execute() {
            for (int i = 0; i < Owner.Neurons; i++) {
                for (int j = 0; j < Owner.Neurons; j++) {
                    Owner.EdgeInputs.Host[i * Owner.Neurons + j] = 0;
                }
                Owner.ImageInput.Host[i] = 0;
                Owner.InnerStates.Host[i] = 0;
            }

            random();

            Owner.ImageInput.SafeCopyToDevice();
            Owner.ImageOutput.SafeCopyToDevice();
            Owner.EdgeInputs.SafeCopyToDevice();
            Owner.Weights.SafeCopyToDevice();
            Owner.InnerStates.SafeCopyToDevice();
            Owner.OutputsIndex.SafeCopyToDevice();
        }

        private void random() {
            Random rand = new Random();

            // Image input randomization

            List<int> tempSet = new List<int>();
            for (int i = 0; i < this.Owner.Input.Count; i++) {
                int temp = rand.Next(0, Owner.Neurons);

                while (tempSet.Contains(temp)) {
                    temp = rand.Next(0, Owner.Neurons);
                }

                tempSet.Add(temp);
                Owner.ImageOutput.Host[i] = temp;
            }

            //Outputs

            int index = 0;
            for (int i = 0; i < this.Owner.Neurons; i++) {
                if (tempSet.Contains(i)) continue;
                Owner.OutputsIndex.Host[index] = i;
                index++;
            }

            // Edges randomization

            int neighbours = Convert.ToInt32(Owner.Neurons * Owner.Connectivity);

            for (int i = 0; i < Owner.Neurons; i++) {
                for (int j = 0; j < Owner.Neurons; j++) {
                    Owner.Weights.Host[i * Owner.Neurons + j] = 0;
                }
            }

            for (int i = 0; i < Owner.Neurons; i++) {
                tempSet = new List<int>();
                for (int j = 0; j < neighbours; j++) {
                    int temp = rand.Next(0, Owner.Neurons);

                    while (temp == i || tempSet.Contains(temp)) {
                        temp = rand.Next(0, Owner.Neurons);
                    }

                    tempSet.Add(temp);

                    float weight = rand.Next(1, 100) / 100.0f;

                    switch (ConnectivityType) {
                        case IOEnum.input:
                            Owner.Weights.Host[temp * Owner.Neurons + i] = weight;
                            break;
                        default:
                            Owner.Weights.Host[i * Owner.Neurons + temp] = weight;
                            break;
                    }
                }
            }
        }

        public int getNeurons() {
            return Owner.Neurons = this.Neurons;
        }
    }
}

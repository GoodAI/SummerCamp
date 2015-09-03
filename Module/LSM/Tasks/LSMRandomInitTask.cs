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
    [Description("Init random network"), MyTaskInfo(OneShot = true)]
    class LSMRandomInitTask : MyTask<LiquidStateMachine> {

        public enum NeuronTypeEnum {
            IF
        }

        [YAXSerializableField(DefaultValue = NeuronTypeEnum.IF)]
        [MyBrowsable, Category("\tLayer")]
        public virtual NeuronTypeEnum NeuronType { get; set; }

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
        }

        private void random() {
            Random rand = new Random();

            List<int> tempSet = new List<int>();
            for (int i = 0; i < this.Owner.Input.Count; i++) {
                int temp = rand.Next(0, Owner.Neurons);

                while (tempSet.Contains(temp)) {
                    temp = rand.Next(0, Owner.Neurons);
                }

                tempSet.Add(temp);
                Owner.ImageOutput.Host[i] = temp;
            }

            //Edges

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

                    Owner.Weights.Host[i * Owner.Neurons + temp] = weight;
                }
            }
        }

        public int getNeurons() {
            return Owner.Neurons = this.Neurons;
        }
    }
}

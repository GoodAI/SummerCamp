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
    /// <summary>Task for initialization of network with a topology according to prof. Maass</summary>
    /// <description>
    /// Starting initialization for LSM with topology according to prof. Maass:<br></br>
    /// - 3D topology with the first depth layer used as input
    /// - neighbours generated randomly with favouring of close neighbours
    /// </description>
    [Description("Init Maass network"), MyTaskInfo(OneShot = true)]
    class LSMMaassInitTask : MyTask<LiquidStateMachine> {

        // constant for calculating of close neighbours
        public const float MAASS_LAMBDA = 2;

        public const double LAMBDA = 0.00000001;

        public enum NeuronTypeEnum {
            IF
        }

        [YAXSerializableField(DefaultValue = NeuronTypeEnum.IF)]
        [MyBrowsable, Category("\tLayer")]
        public virtual NeuronTypeEnum NeuronType { get; set; }

        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int MaassDepth { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float maassC { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLayer")]
        public virtual bool maassBackDepthEdges { get; set; }

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

            maass();
            
            Owner.ImageInput.SafeCopyToDevice();
            Owner.ImageOutput.SafeCopyToDevice();
            Owner.EdgeInputs.SafeCopyToDevice();
            Owner.Weights.SafeCopyToDevice();
            Owner.InnerStates.SafeCopyToDevice();
        }


        private void maass() {
            int[] dimensions = new int[3];
            int[] tempDim = getRectangle(Owner.Input.Count);

            dimensions[0] = tempDim[0];
            dimensions[1] = tempDim[1];
            dimensions[2] = this.MaassDepth;

            // Setting of input neurons
            List<int> tempSet = new List<int>();
            for (int i = 0; i < this.Owner.Input.Count; i++) {
                Owner.ImageOutput.Host[i] = i;
            }

            // Edges randomization
            for (int i = 0; i < Owner.Neurons; i++) {
                for (int j = 0; j < Owner.Neurons; j++) {
                    Owner.Weights.Host[i * Owner.Neurons + j] = 0;
                }
            }


            Random rand = new Random();
            for (int i = 0; i < Owner.Neurons; i++) {
                int[] aDim = new int[] { i % dimensions[1], i / dimensions[1], i / Owner.Inputs };
                int neighbours = Convert.ToInt32((Owner.Neurons - aDim[2] * Owner.Inputs) * Owner.Connectivity);
                int[] nPerm = getPermutation(Owner.Neurons);
                int nCount = 0;
                int index = 0;
                tempSet = new List<int>();
                while (nCount < neighbours) {
                    if (index >= Owner.Neurons) {
                        index = 0;
                    }
                    int j = nPerm[index++];

                    if (i != j && !tempSet.Contains(j)) {
                        int[] bDim = new int[] { j % dimensions[1], j / dimensions[1], j / Owner.Inputs };

                        if (this.maassBackDepthEdges || aDim[2] <= bDim[2]) {
                            double probability = euclideanDistance(aDim, bDim);
                            probability = this.maassC * Math.Exp(-Math.Pow(probability / LSMMaassInitTask.MAASS_LAMBDA, 2));

                            if (probability < 1 && probability >= rand.NextDouble()) {
                                float weight = rand.Next(1, 100) / 100.0f;
                                Owner.Weights.Host[i * Owner.Neurons + j] = weight;

                                tempSet.Add(j);
                                nCount++;
                            }
                        }
                    }
                }
            }
        }

        //Compute sides of rectangle of n neurons
        private int[] getRectangle(int n) {
            int a = Convert.ToInt32(Math.Floor(Math.Sqrt(n)+LAMBDA));
            while (n % a != 0) {
                a--;
            }
            return new int[] {n/a, a};
        }

        // Generating permutation of int array 0..n-1
        private int[] getPermutation(int n) {
            int[] permutation = new int[n];
            for (int i = 0; i < n; i++) {
                permutation[i] = i;
            }

            Random rand = new Random();

            for (int i = n; i > 0; i--) {
                int j = rand.Next(i);

                int temp = permutation[j];
                permutation[j] = permutation[i - 1];
                permutation[i - 1] = temp;
            }

            return permutation;
        }

        // Calculating enclidean distance of neurons
        private double euclideanDistance(int[] a, int[] b) {
            double dist = 0;

            for (int dim = 0; dim < a.Length; dim++) {
                dist += Math.Pow(a[dim]-b[dim],2);
            }


            return Math.Sqrt(dist);
        }

        public int getNeurons() {
            return Owner.Neurons = Owner.Inputs * this.MaassDepth;
        }
    }


}

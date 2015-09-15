using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Windows.Forms.Design;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using System.Drawing.Design;
using System.ComponentModel.Design;
using GoodAI.Core.Task;
using System.IO;
using System.Threading;
using System.Globalization;

namespace LSMModule {
    /// <author>Adr33</author>
    /// <meta>ok</meta>
    /// <status>Work in progress</status>
    /// <summary>LSM Benchmark File Saver node</summary>
    /// <description>
    /// Modification of MyCsvFileWriterNode, which serves for benchmark testing of performance of LSM<br></br>
    /// - the value which is tracked is fault of output, which is difference between label and output
    ///   - the fault is calculated as sum over the elements of absolute value of substraction between output and label matrices
    ///   - the fault is always calculated over the whole input block(whole file, input set)
    /// - the basic setting of LSM and the environment is also saved alongside the fault
    /// - this node serves purely for testing purpose, it is not needed for the run of LSM
    /// </description>
    class BenchmarkFileSaver2 : MyWorkingNode {

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputDirectory"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(FolderNameEditor), typeof(UITypeEditor))]
        public string OutputDirectory { get; set; }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputFile.csv"), YAXElementFor("Structure")]
        public string OutputFile { get; set; }

        [YAXSerializableField(DefaultValue = 100)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int BlockSize { get; set; }

        private StreamWriter m_stream;

        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Target {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Output {
            get { return GetInput(1); }
        }

        #endregion


        public MyWriterTask2 MainTask { get; protected set; }



        // TASKS -----------------------------------------------------------
        [Description("Write Row"), MyTaskInfo(OneShot = false)]
        public class MyWriterTask2 : MyTask<BenchmarkFileSaver2> {
            StreamWriter m_stream;
            int m_count;
            int m_iter;
            float m_sum;

            public override void Init(int nGPU) {

                if (Owner.m_stream != null) {
                    Owner.m_stream.Close();
                }

                Owner.m_stream = new StreamWriter(Owner.OutputDirectory + '\\' + Owner.OutputFile, true);

                m_stream = Owner.m_stream;

                m_count = 0;
                m_iter = 0;

                m_stream.WriteLine("");
                m_stream.Flush();
            }

            public override void Execute() {
                // Saves the fault of current step to temporary memory
                // If end of block is reached saves the average fault over the block to the file
                if ((Owner.Output != null)) {
                    Owner.Output.SafeCopyToHost();
                    Owner.Target.SafeCopyToHost();

                    int top = -1;

                    for (int i = 0; i < 10; i++) {
                        if (Owner.Target.Host[i] > 0.5f) {
                            top = i;
                            break;
                        }
                    }

                    bool good = true;
                    float max = Owner.Output.Host[top];

                    for (int i = 0; i < 10; i++) {
                        if (Owner.Output.Host[i] > max) {
                            good = false;
                            break;
                        }
                    }

                    if (good) {
                        m_count++;
                    }

                    m_iter++;

                    if (m_iter >= Owner.BlockSize) {
                        StringBuilder sb = new StringBuilder();

                        sb.Append(m_count.ToString());
                        sb.Append(';');

                        m_stream.Write(sb.ToString());
                        m_stream.Flush();

                        m_count = 0;
                        m_iter = 0;
                    }
                }
            }
        }


        public override void UpdateMemoryBlocks() {
        }

        public override void Validate(MyValidator validator) {
            validator.AssertError(Directory.Exists(OutputDirectory), this, "The output directory does not exist.");
        }

    }
}

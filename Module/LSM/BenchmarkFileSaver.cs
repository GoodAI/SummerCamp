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
    /// <status>Alpha release</status>
    /// <summary>LSM Benchmark File Saver node</summary>
    /// <description>
    /// Modification of MyCsvFileWriterNode, which serves for benchmark testing of performance of LSM<br></br>
    /// - the value which is tracked is whether detector was able to correctly recognize input as target<br></br>
    ///   - detector recognized input as target if the correct value in target has the biggest score in output<br></br>
    ///   - the value is calculated over the whole input block(whole file, input set) and then divided by the size of block<br></br>
    ///   - so the final value equal to how many percents of inputs over the block were correctly recognized<br></br>
    /// - this node serves purely for testing purpose, it is not needed for the run of LSM
    /// </description>
    class BenchmarkFileSaver : MyWorkingNode {

        public enum FileWriteMethod {
            Overwrite,
            Append
        }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputDirectory"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(FolderNameEditor), typeof(UITypeEditor))]
        public string OutputDirectory { get; set; }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = "outputFile.csv"), YAXElementFor("Structure")]
        public string OutputFile { get; set; }

        [MyBrowsable, Category("\t Output")]
        [YAXSerializableField(DefaultValue = FileWriteMethod.Overwrite)]
        public FileWriteMethod WriteMethod { get; set; }

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


        public MyWriterTask MainTask { get; protected set; }



        // TASKS -----------------------------------------------------------
        [Description("Write Row"), MyTaskInfo(OneShot = false)]
        public class MyWriterTask : MyTask<BenchmarkFileSaver> {
            StreamWriter m_stream;
            int m_count;
            int m_iter;

            public override void Init(int nGPU) {

                if (Owner.m_stream != null) {
                    Owner.m_stream.Close();
                }

                bool append = (Owner.WriteMethod == FileWriteMethod.Append) ? true : false;
                Owner.m_stream = new StreamWriter(Owner.OutputDirectory + '\\' + Owner.OutputFile, append);

                m_stream = Owner.m_stream;

                m_count = 0;
                m_iter = 0;

                if (Owner.WriteMethod == FileWriteMethod.Overwrite) {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < 1001; i++) {
                        sb.Append(i.ToString());
                        sb.Append(';');
                    }

                    m_stream.Write(sb);
                    m_stream.Flush();
                }

                m_stream.WriteLine("");
                m_stream.Flush();
            }

            public override void Execute() {
                // Calculates whether input was recognized correctly
                // If end of block is reached saves the percentage of correct guesses over the whole block
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
                        if (i != top && Owner.Output.Host[i] > max) {
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

                        float temp = m_count;
                        temp /= m_iter;
                        temp *= 100;

                        sb.Append(temp.ToString("0.00"));
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

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

        [MyBrowsable, Category("Content")]
        [YAXSerializableField(DefaultValue = "trainrate;neurons;inputs;connectivity;threshold;spikes;A;B;blocksize"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(MultilineStringEditor), typeof(UITypeEditor))]
        public string Headers { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputSize { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputWidth { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1u), YAXElementFor("Structure")]
        public uint InputHeight { get; private set; }

        [YAXSerializableField(DefaultValue = 0.0025f)]
        [MyBrowsable, Category("\tLayer")]
        public virtual float Trainrate { get; set; }

        [YAXSerializableField(DefaultValue = 400)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int Neurons { get; set; }

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

        private StreamWriter m_stream;

        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Fault {
            get { return GetInput(0); }
        }

        #endregion


        public MyWriterTask MainTask { get; protected set; }



        // TASKS -----------------------------------------------------------
        [Description("Write Row"), MyTaskInfo(OneShot = false)]
        public class MyWriterTask : MyTask<BenchmarkFileSaver> {
            StreamWriter m_stream;
            int m_count;
            int m_iter;
            float m_sum;

            public override void Init(int nGPU) {

                if (Owner.m_stream != null) {
                    Owner.m_stream.Close();
                }

                bool append = (Owner.WriteMethod == FileWriteMethod.Append) ? true : false;
                Owner.m_stream = new StreamWriter(Owner.OutputDirectory + '\\' + Owner.OutputFile, append);

                m_stream = Owner.m_stream;

                m_count = 0;
                m_iter = 0;
                m_sum = 0;

                // When appending, dont add the headers
                if (!String.IsNullOrEmpty(Owner.Headers) && (Owner.WriteMethod == FileWriteMethod.Overwrite)) {
                    StringBuilder sb = new StringBuilder();
                    if (Owner.Headers.EndsWith(Environment.NewLine)) {
                        Owner.Headers.Remove(Owner.Headers.Length - 1);
                    }
                    sb.Append(Owner.Headers);

                    for (int i = 0; i < 1001; i++) {
                        sb.Append(i.ToString());
                        sb.Append(';');
                    }

                    m_stream.Write(sb);
                    m_stream.Flush();
                }

                m_stream.WriteLine();

                // Writing of basic LSM info

                StringBuilder sb2 = new StringBuilder();

                sb2.Append(Owner.Trainrate.ToString("0.0000"));
                sb2.Append(';');

                sb2.Append(Owner.Neurons);
                sb2.Append(';');

                sb2.Append(Owner.Inputs);
                sb2.Append(';');

                sb2.Append(Owner.Connectivity.ToString("0.000"));
                sb2.Append(';');

                sb2.Append(Owner.Threshhold.ToString("0.00"));
                sb2.Append(';');

                sb2.Append(Owner.Spikes.ToString());
                sb2.Append(';');

                sb2.Append(Owner.A.ToString("0.000"));
                sb2.Append(';');

                sb2.Append(Owner.B.ToString("0.000"));
                sb2.Append(';');

                sb2.Append(Owner.BlockSize.ToString());
                sb2.Append(';');

                m_stream.Write(sb2);
                m_stream.Flush();
            }

            public override void Execute() {
                // Saves the fault of current step to temporary memory
                // If end of block is reached saves the average fault over the block to the file
                if ((Owner.Fault != null)) {
                    Owner.Fault.SafeCopyToHost();

                    m_count++;
                    m_sum += Owner.Fault.Host[0];

                    if (m_count >= Owner.BlockSize) {
                        m_iter++;
                        float fault = m_sum / m_count;
                        m_sum = 0;

                        StringBuilder sb = new StringBuilder();

                        sb.Append(fault.ToString("0.00"));
                        sb.Append(';');

                        m_stream.Write(sb.ToString());
                        m_stream.Flush();

                        m_count = 0;
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

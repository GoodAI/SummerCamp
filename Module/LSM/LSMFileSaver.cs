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

namespace LSMModule{
    /// <author>Adr33</author>
    /// <meta>mv</meta>
    /// <status>Work in progress</status>
    /// <summary>LSM File Saver node</summary>
    /// <description>TBA</description>
    class lsmFileSaver : MyWorkingNode {

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

        [YAXSerializableField(DefaultValue = 200)]
        [MyBrowsable, Category("\tLayer")]
        public virtual int BlockSize { get; set; }

        [MyBrowsable, Category("Content")]
        [YAXSerializableField(DefaultValue = "timestamp,iteration,neurons,inputs,connectivity,threshold,spikes,A,B,blocksize,averageFault"), YAXElementFor("Structure")]
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

        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input {
            get { return GetInput(0); }
        }

        #endregion


        public MyWriterTask SpTask { get; protected set; }



        // TASKS -----------------------------------------------------------
        [Description("Write Row"), MyTaskInfo(OneShot = false)]
        public class MyWriterTask : MyTask<lsmFileSaver> {
            StreamWriter m_stream;
            int m_step;
            int m_count;
            int m_iter;
            float m_sum;

            public override void Init(int nGPU) {
                bool append = (Owner.WriteMethod == FileWriteMethod.Append) ? true : false;
                m_stream = new StreamWriter(Owner.OutputDirectory + '\\' + Owner.OutputFile, append);
                m_step = 0;
                m_count = 0;
                m_iter = 0;
                m_sum = 0;

                // when appending, dont add the headers
                if (!String.IsNullOrEmpty(Owner.Headers) && (Owner.WriteMethod == FileWriteMethod.Overwrite)) {
                    StringBuilder sb = new StringBuilder();
                    if (Owner.Headers.EndsWith(Environment.NewLine)) {
                        Owner.Headers.Remove(Owner.Headers.Length - 1);
                    }
                    sb.Append(Owner.Headers);
                    m_stream.WriteLine(sb);
                    m_stream.Flush();
                }
            }

            public override void Execute() {
                m_step++;
                if ((Owner.Input != null)) {
                    Owner.Input.SafeCopyToHost();

                    m_count++;
                    m_sum += Owner.Input.Host[Owner.Input.Count - 1];

                    if (m_count >= Owner.BlockSize) {
                        m_iter++;
                        float fault = m_sum / m_count;
                        m_sum = 0;

                        StringBuilder sb = new StringBuilder();

                        sb.Append(m_step);
                        sb.Append(',');

                        sb.Append(m_iter);
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[0].ToString("0"));
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[1].ToString("0"));
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[2].ToString("0.000", new CultureInfo("en-US")));
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[3].ToString("0.00", new CultureInfo("en-US")));
                        sb.Append(',');

                        float spikesFloat = Owner.Input.Host[4];
                        if (spikesFloat > 0.5f) {
                            sb.Append("true");
                        } else {
                            sb.Append("false");
                        }
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[5].ToString("0.000", new CultureInfo("en-US")));
                        sb.Append(',');

                        sb.Append(Owner.Input.Host[6].ToString("0.000", new CultureInfo("en-US")));
                        sb.Append(',');

                        sb.Append(m_count);
                        sb.Append(',');

                        sb.Append(fault.ToString("0.00", new CultureInfo("en-US")));
                        sb.Append(',');

                        m_stream.WriteLine(sb.ToString());
                        m_stream.Flush();

                        m_count = 0;
                    }
                }
            }
        }


        public override void UpdateMemoryBlocks() {
            InputSize = Input == null ? 1 : (uint)Input.Count;
            InputWidth = Input == null ? 1 : (uint)Input.ColumnHint;
            InputHeight = (uint)Math.Ceiling((float)InputSize / (float)InputWidth);
        }

        public override void Validate(MyValidator validator) {
            validator.AssertError(Directory.Exists(OutputDirectory), this, "The output directory does not exist.");
        }

    }
}

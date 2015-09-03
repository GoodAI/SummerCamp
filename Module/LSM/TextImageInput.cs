using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.Design;
using YAXLib;

namespace LSMModule {
    /// <author>Adr33</author>
    /// <meta>mv</meta>
    /// <status>Work in progress</status>
    /// <summary>Text image input node node</summary>
    /// <description>TBA</description>
    class TextImageInput : MyWorkingNode {

        public const int LABEL = 5;

        [MyBrowsable, Category("\t Letters")]
        [YAXSerializableField(DefaultValue = "lettersDirectory"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(FolderNameEditor), typeof(UITypeEditor))]
        public string LettersDirectory { get; set; }

        [YAXSerializableField(DefaultValue = 5)]
        [MyBrowsable, Category("\t Letters")]
        public virtual int Width { get; set; }

        [YAXSerializableField(DefaultValue = 7)]
        [MyBrowsable, Category("\t Letters")]
        public virtual int Height { get; set; }

        [MyBrowsable, Category("\t Input")]
        [YAXSerializableField(DefaultValue = "inputFile"), YAXElementFor("Structure")]
        [EditorAttribute(typeof(FileNameEditor), typeof(UITypeEditor))]
        public string InputFile { get; set; }

        private StreamReader m_stream;

        #region Memory blocks

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Label {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        #endregion

        public MyTextImageInputTask MainTask { get; protected set; }

        public override void UpdateMemoryBlocks() {
            Output.Count = Height * Width;
            Output.ColumnHint = Width;

            Label.Count = TextImageInput.LABEL;
        }

        [Description("Main Task"), MyTaskInfo(OneShot = false)]
        public class MyTextImageInputTask : MyTask<TextImageInput> {

            float[, ,] outputs;
            StreamReader m_stream;
            bool space;

            public override void Init(int nGPU) {
                outputs = new float[27, Owner.Height, Owner.Width];

                using (TextReader reader = File.OpenText(Owner.LettersDirectory + '\\' + "space.txt")) {
                    for (int j = 0; j < Owner.Height; j++) {
                        string text = reader.ReadLine();
                        string[] bits = text.Split(' ');
                        for (int k = 0; k < Owner.Width; k++) {
                            outputs[0, j, k] = int.Parse(bits[k]); ;
                        }
                    }
                }

                for (char c = 'a'; c <= 'z'; c++) {
                    int i = Convert.ToInt32(c) - Convert.ToInt32('a') + 1;
                    using (TextReader reader = File.OpenText(Owner.LettersDirectory + '\\' + c + ".txt")) {
                        for (int j = 0; j < Owner.Height; j++) {
                            string text = reader.ReadLine();
                            string[] bits = text.Split(' ');
                            for (int k = 0; k < Owner.Width; k++) {
                                outputs[i, j, k] = int.Parse(bits[k]); ;
                            }
                        }
                    }
                }

                if (Owner.m_stream != null) {
                    Owner.m_stream.Close();
                }

                Owner.m_stream = new StreamReader(Owner.InputFile);

                m_stream = Owner.m_stream;

                space = false;
            }

            private char getNext() {
                if (m_stream.EndOfStream) {
                    Owner.m_stream.Close();
                    Owner.m_stream = new StreamReader(Owner.InputFile);

                    m_stream = Owner.m_stream;
                }

                char c = (char)m_stream.Read();
                
                if(c >= 'A' || c <= 'Z'){
                    c = Char.ToLower(c);
                }

                return c;
            }

            public override void Execute() {
                char c = getNext();

                int index = Convert.ToInt32(c) - Convert.ToInt32('a') + 1;

                if (index < 1 || index > 27) {
                    if (space) {
                        while (index < 1 || index > 27) {
                            c = getNext();
                            index = Convert.ToInt32(c) - Convert.ToInt32('a') + 1;
                        }
                        space = false;
                    } else {
                        index = 0;
                        space = true;
                    }
                } else {
                    space = false;
                }

                for (int i = 0; i < Owner.Height; i++) {
                    for (int j = 0; j < Owner.Width; j++) {
                        Owner.Output.Host[i * Owner.Width + j] = outputs[index, i, j];
                    }
                }

                int mod = Convert.ToInt32(Math.Pow(2, LABEL - 1));
                int temp = index;
                for (int i = 0; i < LABEL; i++) {
                    if (temp >= mod) {
                        Owner.Label.Host[i] = 1.0f;
                        temp -= mod;
                    } else {
                        Owner.Label.Host[i] = 0;
                    }
                    mod /= 2;
                }

                Owner.Label.SafeCopyToDevice();
                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}

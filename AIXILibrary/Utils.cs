using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    //public class MyRandom
    //{
    //    public int a = 17;
    //    public int b = 11;
    //    public int x = 51;
    //    public int modulo = 10000;
    //    public int NextNumber() {
    //        x = (x * a + b) % modulo;

    //        return x;
    //    }

    //    public int NextInt(int min=0, int max=10) {
    //        int length = max - min;
    //        return (NextNumber() % length) + min;
    //    }


    //    public double NextDouble() {
    //        return (((double)NextNumber()) / NextNumber()) % 1;
    //    }
    //}
    public class Utils
    {
        public static Random Rnd = new Random();

        public static bool FloatCompare(double a, double b, double delta=1e-5) {
            return Math.Abs(a - b) < delta;
        }
        
        static public int LogBase2(int value)
        {//Copied from internet, probably stack overflow, I forgot where.
            int log = 31;
            while (log >= 0)
            {
                uint mask = ((uint)1 << log);
                if ((mask & value) != 0)
                    return (int)log;
                log--;
            }
            return -1;
        }

        static public int BitsNeeded(int value) {
            return LogBase2(value) + 1;
        }

        static public double RandomDouble(double min, double max)
        {
            return Rnd.NextDouble() * (max - min) + min;
        }

        static public bool ProbabilisticDecision(double limit) {
            return Rnd.NextDouble() < limit;
        }

        static public double MyToDouble(string s) {
            var style = NumberStyles.AllowDecimalPoint;
            var culture = CultureInfo.CreateSpecificCulture("en-US");

            double res;

            Double.TryParse(s, style, culture, out res);
            return res;
        }

        static public int RandomElement(int[] a) {//TODO: int->any type.
            return a[Rnd.Next(a.Length)];//todo: not a.Length-1?
        }

        static public double Log1P(double x) {
        //Copied from John D Cook, licence: public domain
            // http://www.johndcook.com/blog/csharp_log_one_plus_x/
            if (x <= -1.0)
            {
                string msg = String.Format("Invalid input argument: {0}", x);
                throw new ArgumentOutOfRangeException(msg);
            }

            if (Math.Abs(x) > 1e-4)
            {
                // x is large enough that the obvious evaluation is OK
                return Math.Log(1.0 + x);
            }

            // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3
            // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8

            return (-0.5*x + 1.0)*x;
        }

        public static int[] Encode(int integerSymbol, int bitCount) { 
            string s = Convert.ToString(integerSymbol, 2);
            int[] symbolList = s.PadLeft(bitCount, '0') // Add 0's from left
             .Select(c => int.Parse(c.ToString())) // convert each char to int
             .ToArray();
            return symbolList;
        }

        public static int Decode(int[] symbolList, int bitCount) {
            Debug.Assert(bitCount>0 && bitCount <= symbolList.Length);
            
            int value = 0;
            for (int i = 0; i < bitCount; i++)
            {
                if (symbolList[symbolList.Length -i -1] == 1)
                {
                    value += Convert.ToInt32(Math.Pow(2, bitCount -i -1));
                }
            }
            return value;
        }

        static public string PrettyPrintArray(string[] a) {
            return ":-)";//TODO
        }

        static public string IntArrayToString(int[] a)
        {
            return string.Join(", ", a.Select(v => v.ToString()));

        }


        //static public int? ArrayMax(int[] a) {
        //    if a.Length>0{
        //        int max=a[1];
        //    }
        //    foreach (int i in a) {
        //        max = Math.Max(i, max);
        //    }
        //    return max;
        //}

        static public void UnivLogger(string s) {
            Console.WriteLine(s);
            Console.Beep();
            
            //todo: MyLog.INFO.WriteLine();
            //todo: MyLog.Writer.flush()
        }
    }
}

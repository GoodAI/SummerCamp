using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
//using GoodAI.Core.Utils;

namespace AIXI
{
    public class TttEnvironment : AIXIEnvironment
    {

        public TttEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            this.ValidActions = new[] { 0,1,2,3,4,5,6,7,8};
            this.ValidObservations = new int [174762+1]; // This one may be wrong.
            for (int k = 0; k < ValidObservations.Length; k++) {
                this.ValidObservations[k] = k;
            }
                this.ValidRewards = new[] { 0, 1, 2, 3, 4 };
            this.Reward = 0;
            base.fill_out_bits();


        }

        public override Tuple<int, int> PerformAction(int action) {
            this.Action = action;
            //MyLog.INFO.WriteLine("Perf action in TTT: "+action);
            return new Tuple<int, int>(42, 84);
        }
    }
}

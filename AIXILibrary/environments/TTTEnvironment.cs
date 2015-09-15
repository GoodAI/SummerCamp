using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
//using GoodAI.Core.Utils;

namespace AIXI
{
    public class TTTEnvironment : AIXIEnvironment
    {

        public TTTEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            this.valid_actions = new int[9] { 0,1,2,3,4,5,6,7,8};
            this.valid_observations = new int [174762+1]; // This one may be wrong.
            for (int k = 0; k < valid_observations.Length; k++) {
                this.valid_observations[k] = k;
            }
                this.valid_rewards = new int[5] { 0, 1, 2, 3, 4 };
            this.reward = 0;

        }

        public override Tuple<int, int> performAction(int action) {
            this.action = action;
            //MyLog.INFO.WriteLine("Perf action in TTT: "+action);
            return new Tuple<int, int>(42, 84);
        }
    }
}

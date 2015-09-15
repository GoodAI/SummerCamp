using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class CTWContextTreeUndo
    {
        public int age;
        public double total_reward;
        public int history_size;
        public int last_update;

        public CTWContextTreeUndo(MC_AIXI_CTW agent)
        {
            this.age = agent.age;
            this.total_reward = agent.total_reward;
            this.history_size = agent.history_size();
            this.last_update = agent.last_update;
        }
    }
}

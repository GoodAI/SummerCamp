using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class MazeEnvironment : AIXIEnvironment
    {
        public enum maze_action_enum { aLeft, aUp, aRight, aDown };

        public enum maze_observation_enum { oNull = 0, oLeftWall = 1, oUpWall = 2, oRightWall = 4, oDownWall = 8 };

        public enum maze_observation_encoding_enum { cUninformative, cWalls, cCoordinates};//not implemented

        public int aLeft = (int)maze_action_enum.aLeft;
        public int aUp = (int)maze_action_enum.aUp;
        public int aRight = (int)maze_action_enum.aRight;
        public int aDown = (int)maze_action_enum.aDown;

        public int oNull = (int)maze_observation_enum.oNull;
        public int oLeftWall = (int)maze_observation_enum.oLeftWall;
        public int oUpWall = (int)maze_observation_enum.oUpWall;
        public int oRightWall = (int)maze_observation_enum.oRightWall;
        public int oDownWall = (int)maze_observation_enum.oDownWall;

        //public int cUninformative = (int)maze_observation_encoding_enum.cUninformative;
        //public int cWalls = (int)maze_observation_encoding_enum.cWalls;
        //public int cCoordinates = (int)maze_observation_encoding_enum.cCoordinates;

        public char cWall = '#';
        public char cEmpty = '.';

        public char cTeleportTo = '*'; //not implemented
        public char cTeleportFrom = '!'; //not implemented

        public int x;
        public int y;

        public int width;
        public int height;

        public char[,] maze;
        public int[,] rewards;

        public int min_reward_unnormalized;
        public int max_reward_unnormalized;

        public int outside_maze_reward = 0;    //this is assumed to be already normalized to 0+

        public MazeEnvironment(Dictionary<string, string> options, string layout, int[,] rewardsNotNormalized)
            : base(options)
        {
            this.valid_actions = new int[] { this.aLeft, this.aUp, this.aRight, this.aDown };
            this.valid_observations= new int[this.max_observation()+1];
            for (int i = 0; i < this.max_observation() + 1; i++)
            {
                this.valid_observations[i] = i;
            }
            //valid_rewards are defined bellow
            

            var rows = layout.Split(new string[] { Environment.NewLine }, StringSplitOptions.None);
            this.height = rows.Count();

            if (height == 0) {
                throw new ArgumentException("Layout is empty");
            }


            this.width = rows[0].Length;
            this.maze = new char[height, width];

            if (rewardsNotNormalized.GetLength(0) != height || rewardsNotNormalized.GetLength(1) != width)
            {
                throw new ArgumentException("Rewards have different shape than layout");
            }

            this.rewards = rewardsNotNormalized;

            //normalizing rewards:
            this.min_reward_unnormalized=int.MaxValue;
            this.max_reward_unnormalized = int.MinValue;
            foreach (int value in rewardsNotNormalized)
            {
                min_reward_unnormalized = Math.Min(min_reward_unnormalized, value);
                max_reward_unnormalized = Math.Max(max_reward_unnormalized, value);
            }

            this.valid_rewards = new int[(this.max_reward_unnormalized-this.min_reward_unnormalized) + 1];
            for (int i = 0; i < (this.max_reward_unnormalized - this.min_reward_unnormalized) + 1; i++)
            {
                this.valid_rewards[i] = i;
            }

            for (int y = 0; y < height; y++)
            {
                string row = rows[y];
                if (row.Length != width)
                {
                    throw new ArgumentException("maze is not rectangular");
                }
                for (int x = 0; x < width; x++)
                {
                    this.maze[y, x] = row[x];
                    //normalizing rewards to being positive
                    this.rewards[y, x] = rewardsNotNormalized[y,x] - min_reward_unnormalized;
                }
            }
                    

            this.place_agent();

            this.calculate_observation();
            this.reward = 0;
        }

        public bool exists_free_space() {
            foreach (var ch in this.maze) {
                if (ch == cEmpty) {
                    return true;
                }
            }
            return false;
        }

        public void place_agent() {
            if (!this.exists_free_space()) {
                throw new ArgumentException("There is no free spot in maze");
            }
            do {
                this.x = Utils.rnd.Next(0, width);
                this.y = Utils.rnd.Next(0, height);
            } while (!this.accessible(this.x,this.y));
            
        }

        public bool accessible(int x, int y) {
            return this.getPosition(x,y) == this.cEmpty;
        }

        public int max_observation() {
            return this.oLeftWall | this.oRightWall | this.oUpWall | this.oDownWall;
        }

        public void calculate_observation() {
            this.observation = 0;
            if (this.getPosition(x+1, y) == cWall) {
                this.observation |= this.oRightWall;
            }
            if (this.getPosition(x - 1, y) == cWall)
            {
                this.observation |= this.oLeftWall;
            }
            if (this.getPosition(x, y+1) == cWall)
            {
                this.observation |= this.oDownWall;
            }
            if (this.getPosition(x, y-1) == cWall)
            {
                this.observation |= this.oUpWall;
            }
        }

        public int getPosition(int x, int y) {
            if (inMaze(x, y))
            {
                return this.maze[y, x];
            }
            else {
                return cWall;
            }
        }

        public int getReward(int x, int y) {
            if (this.inMaze(x, y)) {
                return this.rewards[y, x];
            }
            return this.outside_maze_reward;
        }

        public bool inMaze(int x, int y) {
            return (x >= 0 && x < width) && (y >= 0 && y < height);
        }

        public int xdiff(int action) {
            if (action == aLeft)
                return -1;
            else if (action == aRight)
                return 1;
            return 0;
        }

        public int ydiff(int action)
        {
            if (action == aUp)
                return -1;
            else if (action == aDown)
                return 1;
            return 0;
        }

        public override Tuple<int, int> performAction(int action)
        {
            int newx = this.x + this.xdiff(action);
            int newy = this.y + this.ydiff(action);

            if (accessible(newx, newy)) {
                this.x = newx;
                this.y = newy;
            }

            this.reward = this.getReward(newx, newy);
            this.calculate_observation();

            return new Tuple<int, int>(this.observation,this.reward);
        }
    }

}

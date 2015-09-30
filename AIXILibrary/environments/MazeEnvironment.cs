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
        public enum MazeActionEnum { ALeft, AUp, ARight, ADown };

        public enum MazeObservationEnum { ONull = 0, OLeftWall = 1, OUpWall = 2, ORightWall = 4, ODownWall = 8 };

        public enum MazeObservationEncodingEnum { CUninformative, CWalls, CCoordinates};//not implemented

        public int ALeft = (int)MazeActionEnum.ALeft;
        public int AUp = (int)MazeActionEnum.AUp;
        public int ARight = (int)MazeActionEnum.ARight;
        public int ADown = (int)MazeActionEnum.ADown;

        public int ONull = (int)MazeObservationEnum.ONull;
        public int OLeftWall = (int)MazeObservationEnum.OLeftWall;
        public int OUpWall = (int)MazeObservationEnum.OUpWall;
        public int ORightWall = (int)MazeObservationEnum.ORightWall;
        public int ODownWall = (int)MazeObservationEnum.ODownWall;

        //public int cUninformative = (int)maze_observation_encoding_enum.cUninformative;
        //public int cWalls = (int)maze_observation_encoding_enum.cWalls;
        //public int cCoordinates = (int)maze_observation_encoding_enum.cCoordinates;

        public char CWall = '#';
        public char CEmpty = '.';

        public char CTeleportTo = '*'; //not implemented
        public char CTeleportFrom = '!'; //not implemented

        public int X;
        public int Y;

        public int Width;
        public int Height;

        public char[,] Maze;
        public int[,] Rewards;

        public int MinRewardUnnormalized;
        public int MaxRewardUnnormalized;

        public int OutsideMazeReward = 0;    //this is assumed to be already normalized to 0+

        public MazeEnvironment(Dictionary<string, string> options, string layout, int[,] rewardsNotNormalized)
            : base(options)
        {
            this.ValidActions = new[] { this.ALeft, this.AUp, this.ARight, this.ADown };
            this.ValidObservations= new int[this.max_observation()+1];
            for (int i = 0; i < this.max_observation() + 1; i++)
            {
                this.ValidObservations[i] = i;
            }
            //valid_rewards are defined bellow
            

            var rows = layout.Split(new[] { Environment.NewLine }, StringSplitOptions.None);
            this.Height = rows.Length;

            if (Height == 0) {
                throw new ArgumentException("Layout is empty");
            }


            this.Width = rows[0].Length;
            this.Maze = new char[Height, Width];

            if (rewardsNotNormalized.GetLength(0) != Height || rewardsNotNormalized.GetLength(1) != Width)
            {
                throw new ArgumentException("Rewards have different shape than layout");
            }

            this.Rewards = rewardsNotNormalized;

            //normalizing rewards:
            this.MinRewardUnnormalized=int.MaxValue;
            this.MaxRewardUnnormalized = int.MinValue;
            foreach (int value in rewardsNotNormalized)
            {
                MinRewardUnnormalized = Math.Min(MinRewardUnnormalized, value);
                MaxRewardUnnormalized = Math.Max(MaxRewardUnnormalized, value);
            }

            this.ValidRewards = new int[(this.MaxRewardUnnormalized-this.MinRewardUnnormalized) + 1];
            for (int i = 0; i < (this.MaxRewardUnnormalized - this.MinRewardUnnormalized) + 1; i++)
            {
                this.ValidRewards[i] = i;
            }

            for (int y = 0; y < Height; y++)
            {
                string row = rows[y];
                if (row.Length != Width)
                {
                    throw new ArgumentException("maze is not rectangular");
                }
                for (int x = 0; x < Width; x++)
                {
                    this.Maze[y, x] = row[x];
                    //normalizing rewards to being positive
                    this.Rewards[y, x] = rewardsNotNormalized[y,x] - MinRewardUnnormalized;
                }
            }
            base.fill_out_bits();


            this.place_agent();

            this.calculate_observation();
            this.Reward = 0;


        }

        public bool exists_free_space() {
            foreach (var ch in this.Maze) {
                if (ch == CEmpty) {
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
                this.X = Utils.Rnd.Next(0, Width);
                this.Y = Utils.Rnd.Next(0, Height);
            } while (!this.Accessible(this.X,this.Y));
            
        }

        public bool Accessible(int x, int y) {
            return this.GetPosition(x,y) == this.CEmpty;
        }

        public int max_observation() {
            return this.OLeftWall | this.ORightWall | this.OUpWall | this.ODownWall;
        }

        public void calculate_observation() {
            this.Observation = 0;
            if (this.GetPosition(X+1, Y) == CWall) {
                this.Observation |= this.ORightWall;
            }
            if (this.GetPosition(X - 1, Y) == CWall)
            {
                this.Observation |= this.OLeftWall;
            }
            if (this.GetPosition(X, Y+1) == CWall)
            {
                this.Observation |= this.ODownWall;
            }
            if (this.GetPosition(X, Y-1) == CWall)
            {
                this.Observation |= this.OUpWall;
            }
        }

        public int GetPosition(int x, int y) {
            if (InMaze(x, y))
            {
                return this.Maze[y, x];
            }
            else {
                return CWall;
            }
        }

        public int GetReward(int x, int y) {
            if (this.InMaze(x, y)) {
                return this.Rewards[y, x];
            }
            return this.OutsideMazeReward;
        }

        public bool InMaze(int x, int y) {
            return (x >= 0 && x < Width) && (y >= 0 && y < Height);
        }

        public int Xdiff(int action) {
            if (action == ALeft)
                return -1;
            else if (action == ARight)
                return 1;
            return 0;
        }

        public int Ydiff(int action)
        {
            if (action == AUp)
                return -1;
            else if (action == ADown)
                return 1;
            return 0;
        }

        public override Tuple<int, int> PerformAction(int action)
        {
            int newx = this.X + this.Xdiff(action);
            int newy = this.Y + this.Ydiff(action);

            if (Accessible(newx, newy)) {
                this.X = newx;
                this.Y = newy;
            }

            this.Reward = this.GetReward(newx, newy);
            this.calculate_observation();

            return new Tuple<int, int>(this.Observation,this.Reward);
        }
    }

}

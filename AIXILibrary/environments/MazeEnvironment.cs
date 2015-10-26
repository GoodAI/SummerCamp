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

        public enum MazeObservationEnum { OLeftWall = 1, OUpWall = 2, ORightWall = 4, ODownWall = 8 };

        //rewards before normalisation:
        // bumping into wall: -10
        // moving to free cell = -1
        // finding cheese=10
        enum RewardEnum { RWall = 0, REmpty= 9, RCheese = 20 };

        public int ALeft = (int)MazeActionEnum.ALeft;
        public int AUp = (int)MazeActionEnum.AUp;
        public int ARight = (int)MazeActionEnum.ARight;
        public int ADown = (int)MazeActionEnum.ADown;

        public int OLeftWall = (int)MazeObservationEnum.OLeftWall;
        public int OUpWall = (int)MazeObservationEnum.OUpWall;
        public int ORightWall = (int)MazeObservationEnum.ORightWall;
        public int ODownWall = (int)MazeObservationEnum.ODownWall;


        public char CWall = '#';
        public char CEmpty = '.';
        public char CCheese = '@';

        public int X;
        public int Y;

        public int Width;
        public int Height;

        public char[,] Maze;

        public int RCheese = (int)RewardEnum.RCheese;
        public int REmpty = (int)RewardEnum.REmpty;
        public int RWall = (int)RewardEnum.RWall;
        public int OutsideMazeReward = (int) RewardEnum.RWall;
        
        public MazeEnvironment(Dictionary<string, string> options, string layout="")
            : base(options)
        {
            //Note: numbering of rows of maze is such:
            // 0 - {first/upper one}
            // 1 - {second one}
            //...
            if (layout == "") {
                layout =
    @"#######
#.....#
#.#.#.#
#.#@#.#
#######";
            }


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


            ValidRewards = (int[])Enum.GetValues(typeof(RewardEnum));


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
            //TODO
            if (this.InMaze(x, y)) {
                if (this.Maze[y,x] == CWall)
                {
                    return this.RWall;
                }
                if (this.Maze[y, x] == CEmpty)
                {
                    return this.REmpty;
                }
                if (this.Maze[y, x] == CCheese)
                {
                    return this.RCheese;
                }


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

            if (this.Reward == this.RCheese)
            {
                place_agent();
            }

            return new Tuple<int, int>(this.Observation,this.Reward);
        }

        public void print()
        {
            for (int y = 0; y < this.Height; y++)
            {
                for (int x = 0; x < this.Width; x++)
                {
                    if (x == this.X && y == this.Y)
                    {
                        Console.Write("A");
                    }
                    else
                    {
                        Console.Write(this.Maze[y, x]);
                    }
                }
                Console.WriteLine();
            }


        }
    }

}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class MyTttEnvironment : AIXIEnvironment
    {
        public enum TictactoeObservationEnum { OEmpty, OAgent, OEnv };    //whose piece is on square

        public enum TictactoeRewardEnum { RInvalid = 0, RLoss = 1, RNull = 3, RDraw = 4, RWin = 5 };

        public int OEmpty = (int)TictactoeObservationEnum.OEmpty;
        public int OAgent = (int)TictactoeObservationEnum.OAgent;
        public int OEnv = (int)TictactoeObservationEnum.OEnv;

        public int RInvalid = (int)TictactoeRewardEnum.RInvalid;
        public int RLoss = (int)TictactoeRewardEnum.RLoss;
        public int RNull = (int)TictactoeRewardEnum.RNull;
        public int RDraw = (int)TictactoeRewardEnum.RDraw;
        public int RWin = (int)TictactoeRewardEnum.RWin;

        public int[,] Board;
        private int _actionsSinceReset;
        public MyTttEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            this.ValidActions=new int[] {0,1,2,3,4,5,6,7,8};
            int maximumPossibleObservation = 174762;//refact: put into hex
            this.ValidObservations = new int[maximumPossibleObservation+1];
            for (int i = 0; i < maximumPossibleObservation+1; i++) {
                this.ValidObservations[i] = i;
            }

            // valid_rewards contains {0,1,3,4,5}
            this.ValidRewards = (int[])Enum.GetValues(typeof(TictactoeRewardEnum));
            base.fill_out_bits();

            this.Reward = 0;
            this.Reset();

        }

        public override Tuple<int, int> PerformAction(int action)
        {
            Debug.Assert(this.IsValidAction(action));

            this.Action = action;

            this._actionsSinceReset+=1;
            int r = action / 3;
            int c = action % 3;

            if (this.Board[r, c] != this.OEmpty) {
                this.Reward = this.RInvalid;
                this.Reset();
                return new Tuple<int,int>(this.Observation, this.Reward);
            }

            this.Board[r, c] = this.OAgent;
            if (this.check_win()) {
                this.Reward = this.RWin;
                this.Reset();
                return new Tuple<int, int>(this.Observation, this.Reward);
            }
            else if (this._actionsSinceReset == 5) {
                this.Reward = this.RDraw;
                this.Reset();
                return new Tuple<int, int>(this.Observation, this.Reward);
            }

            while (this.Board[r, c] != this.OEmpty) {
                r = Utils.Rnd.Next(0, 3);
                c = Utils.Rnd.Next(0, 3);
            }

            this.Board[r, c] = this.OEnv;
            if (this.check_win()) {
                this.Reward = RLoss;
                this.Reset();
                return new Tuple<int, int>(this.Observation, this.Reward);
            }

            this.Reward = RNull;
            this.compute_observation();

            return new Tuple<int, int>(this.Observation, this.Reward);
        }

        public bool check_win() {
            //we do not need to recognize who won: it is player who played last
            for (int r = 0; r < 3; r++) {
                if (this.Board[r,0]!=this.OEmpty &&
                    this.Board[r,0]==this.Board[r,1] &&
                    this.Board[r,1]==this.Board[r,2]
                    ){
                        return true;
                }
            }

            for (int c = 0; c < 3; c++) {
                if (this.Board[0, c] != this.OEmpty &&
                    this.Board[0,c]==this.Board[1,c] &&
                    this.Board[1,c]==this.Board[2,c]
                    ){
                        return true;
                }
            }

            if (this.Board[0,0]!=this.OEmpty &&
                this.Board[0,0]==this.Board[1,1] &&
                this.Board[1,1]==this.Board[2,2]
                ){
                return true;
            }

            if (this.Board[0, 2] != this.OEmpty &&
                this.Board[0, 2] == this.Board[1, 1] &&
                this.Board[1, 1] == this.Board[2, 0]
                )
            {
                return true;
            }
            return false;
        }

        public string Print() { 
            string message = string.Format("action = {0}, observation = {1}, reward = {2} ({3}), board:",
                this.Action,
                this.Observation,
                this.Reward,
                this.Reward-3);

            message = message + Environment.NewLine;

            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    string b=":-( Fix me";
                    if (this.Board[r, c] == OEmpty)
                        b = ".";
                    else if (this.Board[r, c] == OEnv)
                        b = "O";
                    else if (this.Board[r, c] == OAgent)
                        b = "A";
                    else
                        Debug.Assert(false, "on position r/c: " + r + "/" + c + " is wrong value:" + this.Board[r, c]);
                    message += b;
                }
                message += Environment.NewLine;
            }
            message += Environment.NewLine;
            return message;
        }

        public void Reset() { 
            this.Board = new int[3,3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    this.Board[i, j] = OEmpty;
                }
            }

            this.compute_observation();
            this._actionsSinceReset = 0;
        }

        public void compute_observation()
        {
            this.Observation=0;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++ ){
                    this.Observation = this.Board[r,c]+(4*this.Observation);
                }
            }
        }

    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIXI
{
    public class MyTTTEnvironment : AIXIEnvironment
    {
        public enum tictactoe_observation_enum { oEmpty, oAgent, oEnv };    //whose piece is on square

        public enum tictactoe_reward_enum { rInvalid = 0, rLoss = 1, rNull = 3, rDraw = 4, rWin = 5 };

        public int oEmpty = (int)tictactoe_observation_enum.oEmpty;
        public int oAgent = (int)tictactoe_observation_enum.oAgent;
        public int oEnv = (int)tictactoe_observation_enum.oEnv;

        public int rInvalid = (int)tictactoe_reward_enum.rInvalid;
        public int rLoss = (int)tictactoe_reward_enum.rLoss;
        public int rNull = (int)tictactoe_reward_enum.rNull;
        public int rDraw = (int)tictactoe_reward_enum.rDraw;
        public int rWin = (int)tictactoe_reward_enum.rWin;

        public int[,] board;
        private int actions_since_reset;
        public MyTTTEnvironment(Dictionary<string, string> options)
            : base(options)
        {
            this.valid_actions=new int[] {0,1,2,3,4,5,6,7,8};
            int maximum_possible_observation = 174762;//refact: put into hex
            this.valid_observations = new int[maximum_possible_observation+1];
            for (int i = 0; i < maximum_possible_observation+1; i++) {
                this.valid_observations[i] = i;
            }

            // valid_rewards contains {0,1,3,4,5}
            this.valid_rewards = (int[])Enum.GetValues(typeof(tictactoe_reward_enum));

            this.reward = 0;
            this.reset();
        }

        public override Tuple<int, int> performAction(int action)
        {
            Debug.Assert(this.isValidAction(action));

            this.action = action;

            this.actions_since_reset+=1;
            int r = action / 3;
            int c = action % 3;

            if (this.board[r, c] != this.oEmpty) {
                this.reward = this.rInvalid;
                this.reset();
                return new Tuple<int,int>(this.observation, this.reward);
            }

            this.board[r, c] = this.oAgent;
            if (this.check_win()) {
                this.reward = this.rWin;
                this.reset();
                return new Tuple<int, int>(this.observation, this.reward);
            }
            else if (this.actions_since_reset == 5) {
                this.reward = this.rDraw;
                this.reset();
                return new Tuple<int, int>(this.observation, this.reward);
            }

            while (this.board[r, c] != this.oEmpty) {
                r = Utils.rnd.Next(0, 3);
                c = Utils.rnd.Next(0, 3);
            }

            this.board[r, c] = this.oEnv;
            if (this.check_win()) {
                this.reward = rLoss;
                this.reset();
                return new Tuple<int, int>(this.observation, this.reward);
            }

            this.reward = rNull;
            this.compute_observation();

            return new Tuple<int, int>(this.observation, this.reward);
        }

        public bool check_win() {
            //we do not need to recognize who won: it is player who played last
            for (int r = 0; r < 3; r++) {
                if (this.board[r,0]!=this.oEmpty &&
                    this.board[r,0]==this.board[r,1] &&
                    this.board[r,1]==this.board[r,2]
                    ){
                        return true;
                }
            }

            for (int c = 0; c < 3; c++) {
                if (this.board[0, c] != this.oEmpty &&
                    this.board[0,c]==this.board[1,c] &&
                    this.board[1,c]==this.board[2,c]
                    ){
                        return true;
                }
            }

            if (this.board[0,0]!=this.oEmpty &&
                this.board[0,0]==this.board[1,1] &&
                this.board[1,1]==this.board[2,2]
                ){
                return true;
            }

            if (this.board[0, 2] != this.oEmpty &&
                this.board[0, 2] == this.board[1, 1] &&
                this.board[1, 1] == this.board[2, 0]
                )
            {
                return true;
            }
            return false;
        }

        public string print() { 
            string message = string.Format("action = {0}, observation = {1}, reward = {2} ({3}), board:",
                this.action,
                this.observation,
                this.reward,
                this.reward-3);

            message = message + Environment.NewLine;

            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    string b=":-( Fix me";
                    if (this.board[r, c] == oEmpty)
                        b = ".";
                    else if (this.board[r, c] == oEnv)
                        b = "O";
                    else if (this.board[r, c] == oAgent)
                        b = "A";
                    else
                        Debug.Assert(false, "on position r/c: " + r + "/" + c + " is wrong value:" + this.board[r, c]);
                    message += b;
                }
                message += Environment.NewLine;
            }
            message += Environment.NewLine;
            return message;
        }

        public void reset() { 
            this.board = new int[3,3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    this.board[i, j] = oEmpty;
                }
            }

            this.compute_observation();
            this.actions_since_reset = 0;
        }

        public void compute_observation()
        {
            this.observation=0;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++ ){
                    this.observation = this.board[r,c]+(4*this.observation);
                }
            }
        }

    }
}

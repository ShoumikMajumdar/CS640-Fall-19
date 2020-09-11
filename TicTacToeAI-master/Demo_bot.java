import java.util.*;
public class Aaron_bot {

    public int player; //1 for player 1 and 2 for player 2
    public int depth;
    public List<List<positionTicTacToe>> winningLines;

    private int getStateOfPositionFromBoard(positionTicTacToe position, List<positionTicTacToe> board)
    {
        //a helper function to get state of a certain position in the Tic-Tac-Toe board by given position TicTacToe
        int index = position.x*16+position.y*4+position.z;
        return board.get(index).state;
    }

    private List<positionTicTacToe> deepCopyATicTacToeBoard(List<positionTicTacToe> board)
    {
        //deep copy of game boards
        List<positionTicTacToe> copiedBoard = new ArrayList<positionTicTacToe>();
        for(int i=0;i<board.size();i++)
        {
            copiedBoard.add(new positionTicTacToe(board.get(i).x,board.get(i).y,board.get(i).z,board.get(i).state));
        }
        return copiedBoard;
    }

    public boolean terminal(List<positionTicTacToe> board){

        for(int i=0;i<winningLines.size();i++)
        {

            positionTicTacToe p0 = winningLines.get(i).get(0);
            positionTicTacToe p1 = winningLines.get(i).get(1);
            positionTicTacToe p2 = winningLines.get(i).get(2);
            positionTicTacToe p3 = winningLines.get(i).get(3);

            int state0 = getStateOfPositionFromBoard(p0,board);
            int state1 = getStateOfPositionFromBoard(p1,board);
            int state2 = getStateOfPositionFromBoard(p2,board);
            int state3 = getStateOfPositionFromBoard(p3,board);

            //if they have the same state (marked by same player) and they are not all marked.
            if(state0 == state1 && state1 == state2 && state2 == state3 && state0!=0) return true;
        }
        return false;
    }

    public boolean terminal_player(List<positionTicTacToe> board, int p){

        for(int i=0;i<winningLines.size();i++)
        {

            positionTicTacToe p0 = winningLines.get(i).get(0);
            positionTicTacToe p1 = winningLines.get(i).get(1);
            positionTicTacToe p2 = winningLines.get(i).get(2);
            positionTicTacToe p3 = winningLines.get(i).get(3);

            int state0 = getStateOfPositionFromBoard(p0,board);
            int state1 = getStateOfPositionFromBoard(p1,board);
            int state2 = getStateOfPositionFromBoard(p2,board);
            int state3 = getStateOfPositionFromBoard(p3,board);

            //if they have the same state (marked by same player) and they are not all marked.
            if(state0 == state1 && state1 == state2 && state2 == state3 && state0 == p) return true;
        }
        return false;
    }

    public int check(List<positionTicTacToe> board, int p){
        int score = 0;
        Integer scores[] = {10, 100, 500, 10000};
        for(int i = 0;i<winningLines.size();i++){
            positionTicTacToe p0 = winningLines.get(i).get(0);
            positionTicTacToe p1 = winningLines.get(i).get(1);
            positionTicTacToe p2 = winningLines.get(i).get(2);
            positionTicTacToe p3 = winningLines.get(i).get(3);

            int state0 = getStateOfPositionFromBoard(p0,board);
            int state1 = getStateOfPositionFromBoard(p1,board);
            int state2 = getStateOfPositionFromBoard(p2,board);
            int state3 = getStateOfPositionFromBoard(p3,board);

            int cur = 0, other = 0;

            if(state0 == p) cur++;
            else if(state0 != 0) other++;
            if(state1 == p) cur++;
            else if(state1 != 0) other++;
            if(state2 == p) cur++;
            else if(state2 != 0) other++;
            if(state3 == p) cur++;
            else if(state3 != 0) other++;

            if(cur > 0 && other == 0){
                score += scores[cur - 1];
            }
            else if(other > 0 && cur == 0){
                score -= scores[other - 1];
            }
        }
        return score;
    }

    public int heuristic(List<positionTicTacToe> board, int p){
        if(terminal_player(board, p)) return 1000000;
        return check(board, p);
    }

    public int evaluate(List<positionTicTacToe> board, int depth, boolean maximizing, int alpha, int beta, int p){
        if(depth == 0 || terminal(board)) return heuristic(board, p);
        if(maximizing){
            int value = Integer.MAX_VALUE;
            for(int i = 0;i<board.size();i++){
                positionTicTacToe candidate = board.get(i);
                if(getStateOfPositionFromBoard(candidate, board) != 0) continue;
                int nextp = p == 1? 2: 1;
                board.get(candidate.x*16+candidate.y*4+candidate.z).state = nextp;
                value = Integer.max(value, evaluate(board, depth - 1, false, alpha, beta, nextp));
                board.get(candidate.x*16+candidate.y*4+candidate.z).state = 0;
                alpha = Integer.max(alpha, value);
                if(alpha >= beta) break;
            }
            return value;
        }
        else{
            int value = Integer.MAX_VALUE;
            for(int i = 0;i<board.size();i++){
                positionTicTacToe candidate = board.get(i);
                if(getStateOfPositionFromBoard(candidate, board) != 0) continue;
                int nextp = p == 1? 2: 1;
                board.get(candidate.x*16+candidate.y*4+candidate.z).state = nextp;
                value = Integer.min(value, evaluate(board, depth - 1, true, alpha, beta, nextp));
                board.get(candidate.x*16+candidate.y*4+candidate.z).state = 0;
                beta = Integer.min(beta, value);
                if(alpha >= beta) break;
            }
            return value;
        }
    }

    public positionTicTacToe myAIAlgorithm(List<positionTicTacToe> board, int player)
    {
        //TODO: this is where you are going to implement your AI algorithm to win the game. The default is an AI randomly choose any available move.
        positionTicTacToe myNextMove = new positionTicTacToe(0,0,0);
        int value = 0;
        List<positionTicTacToe> copy = deepCopyATicTacToeBoard(board);
        for(int i = 0;i<board.size();i++){
            positionTicTacToe candidate = board.get(i);
            if(getStateOfPositionFromBoard(candidate, board) != 0) continue;
            copy.get(candidate.x*16+candidate.y*4+candidate.z).state = player;
            int x = evaluate(copy, depth, false, Integer.MIN_VALUE, Integer.MAX_VALUE, player);
            copy.get(candidate.x*16+candidate.y*4+candidate.z).state = 0;
            if(x > value){
                value = x;
                myNextMove = candidate;
            }
        }
        return myNextMove;
    }

    private void initializeWinningLines()
    {
        //create a list of winning line so that the game will "brute-force" check if a player satisfied any 	winning condition(s).
        winningLines = new ArrayList<List<positionTicTacToe>>();

        //48 straight winning lines
        //z axis winning lines
        for(int i = 0; i<4; i++)
            for(int j = 0; j<4;j++)
            {
                List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
                oneWinCondtion.add(new positionTicTacToe(i,j,0,-1));
                oneWinCondtion.add(new positionTicTacToe(i,j,1,-1));
                oneWinCondtion.add(new positionTicTacToe(i,j,2,-1));
                oneWinCondtion.add(new positionTicTacToe(i,j,3,-1));
                winningLines.add(oneWinCondtion);
            }
        //y axis winning lines
        for(int i = 0; i<4; i++)
            for(int j = 0; j<4;j++)
            {
                List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
                oneWinCondtion.add(new positionTicTacToe(i,0,j,-1));
                oneWinCondtion.add(new positionTicTacToe(i,1,j,-1));
                oneWinCondtion.add(new positionTicTacToe(i,2,j,-1));
                oneWinCondtion.add(new positionTicTacToe(i,3,j,-1));
                winningLines.add(oneWinCondtion);
            }
        //x axis winning lines
        for(int i = 0; i<4; i++)
            for(int j = 0; j<4;j++)
            {
                List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
                oneWinCondtion.add(new positionTicTacToe(0,i,j,-1));
                oneWinCondtion.add(new positionTicTacToe(1,i,j,-1));
                oneWinCondtion.add(new positionTicTacToe(2,i,j,-1));
                oneWinCondtion.add(new positionTicTacToe(3,i,j,-1));
                winningLines.add(oneWinCondtion);
            }

        //12 main diagonal winning lines
        //xz plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(0,i,0,-1));
            oneWinCondtion.add(new positionTicTacToe(1,i,1,-1));
            oneWinCondtion.add(new positionTicTacToe(2,i,2,-1));
            oneWinCondtion.add(new positionTicTacToe(3,i,3,-1));
            winningLines.add(oneWinCondtion);
        }
        //yz plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(i,0,0,-1));
            oneWinCondtion.add(new positionTicTacToe(i,1,1,-1));
            oneWinCondtion.add(new positionTicTacToe(i,2,2,-1));
            oneWinCondtion.add(new positionTicTacToe(i,3,3,-1));
            winningLines.add(oneWinCondtion);
        }
        //xy plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(0,0,i,-1));
            oneWinCondtion.add(new positionTicTacToe(1,1,i,-1));
            oneWinCondtion.add(new positionTicTacToe(2,2,i,-1));
            oneWinCondtion.add(new positionTicTacToe(3,3,i,-1));
            winningLines.add(oneWinCondtion);
        }

        //12 anti diagonal winning lines
        //xz plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(0,i,3,-1));
            oneWinCondtion.add(new positionTicTacToe(1,i,2,-1));
            oneWinCondtion.add(new positionTicTacToe(2,i,1,-1));
            oneWinCondtion.add(new positionTicTacToe(3,i,0,-1));
            winningLines.add(oneWinCondtion);
        }
        //yz plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(i,0,3,-1));
            oneWinCondtion.add(new positionTicTacToe(i,1,2,-1));
            oneWinCondtion.add(new positionTicTacToe(i,2,1,-1));
            oneWinCondtion.add(new positionTicTacToe(i,3,0,-1));
            winningLines.add(oneWinCondtion);
        }
        //xy plane-4
        for(int i = 0; i<4; i++)
        {
            List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
            oneWinCondtion.add(new positionTicTacToe(0,3,i,-1));
            oneWinCondtion.add(new positionTicTacToe(1,2,i,-1));
            oneWinCondtion.add(new positionTicTacToe(2,1,i,-1));
            oneWinCondtion.add(new positionTicTacToe(3,0,i,-1));
            winningLines.add(oneWinCondtion);
        }

        //4 additional diagonal winning lines
        List<positionTicTacToe> oneWinCondtion = new ArrayList<positionTicTacToe>();
        oneWinCondtion.add(new positionTicTacToe(0,0,0,-1));
        oneWinCondtion.add(new positionTicTacToe(1,1,1,-1));
        oneWinCondtion.add(new positionTicTacToe(2,2,2,-1));
        oneWinCondtion.add(new positionTicTacToe(3,3,3,-1));
        winningLines.add(oneWinCondtion);

        oneWinCondtion = new ArrayList<positionTicTacToe>();
        oneWinCondtion.add(new positionTicTacToe(0,0,3,-1));
        oneWinCondtion.add(new positionTicTacToe(1,1,2,-1));
        oneWinCondtion.add(new positionTicTacToe(2,2,1,-1));
        oneWinCondtion.add(new positionTicTacToe(3,3,0,-1));
        winningLines.add(oneWinCondtion);

        oneWinCondtion = new ArrayList<positionTicTacToe>();
        oneWinCondtion.add(new positionTicTacToe(3,0,0,-1));
        oneWinCondtion.add(new positionTicTacToe(2,1,1,-1));
        oneWinCondtion.add(new positionTicTacToe(1,2,2,-1));
        oneWinCondtion.add(new positionTicTacToe(0,3,3,-1));
        winningLines.add(oneWinCondtion);

        oneWinCondtion = new ArrayList<positionTicTacToe>();
        oneWinCondtion.add(new positionTicTacToe(0,3,0,-1));
        oneWinCondtion.add(new positionTicTacToe(1,2,1,-1));
        oneWinCondtion.add(new positionTicTacToe(2,1,2,-1));
        oneWinCondtion.add(new positionTicTacToe(3,0,3,-1));
        winningLines.add(oneWinCondtion);

    }
    public Aaron_bot(int setPlayer, int d)
    {
        player = setPlayer;
        depth = d;
        initializeWinningLines();
    }
}


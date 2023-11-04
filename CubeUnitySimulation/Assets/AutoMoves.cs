using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AutoMoves : MonoBehaviour
{
    public static Queue<string> movesToDo = new Queue<string>(new[] { "FL'", "BL'", "BR'", "FR'", "T'", "B'", "B", "T", "FR", "BR", "BL", "FL",  });
    private CubeState cubeState;

    // Start is called before the first frame update
    void Start()
    {
        cubeState = FindObjectOfType<CubeState>();
    }

    // Update is called once per frame
    void Update()
    {
        while (movesToDo.Count > 0 && !CubeState.isAutoRotating && CubeState.started)
        {
            string moveToDo = movesToDo.Dequeue();
            doMove(moveToDo);
        }
    }

    void doMove(string move)
    {
        /* 
        T - top
        B - bottom
        FL - front left
        BL - back left
        FR - front right
        BR - back right
        */
        CubeState.isAutoRotating = true;
        if (move == "T" )
        {
            rotateSide(cubeState.top, 90);
        }
        else if (move == "T'" )
        {
            rotateSide(cubeState.top, -90);
        }
        else if (move == "B" )
        {
            rotateSide(cubeState.bottom, 90);
        }
        else if (move == "B'" )
        {
            rotateSide(cubeState.bottom, -90);
        }
        else if (move == "FL" )
        {
            rotateSide(cubeState.frontLeft, 90);
        }
        else if (move == "FL'" )
        {
            rotateSide(cubeState.frontLeft, -90);
        }
        else if (move == "BL" )
        {
            rotateSide(cubeState.backLeft, 90);
        }
        else if (move == "BL'" )
        {
            rotateSide(cubeState.backLeft, -90);
        }
        else if (move == "FR" )
        {
            rotateSide(cubeState.frontRight, 90);
        }
        else if (move == "FR'" )
        {
            rotateSide(cubeState.frontRight, -90);
        }
        else if (move == "BR" )
        {
            rotateSide(cubeState.backRight, 90);
        }
        else if (move == "BR'" )
        {
            rotateSide(cubeState.backRight, -90);
        }
        else {
            print("Invalid move");
        }
    }



    void rotateSide(List<GameObject> side, float angle)
    {
        PivotRotation pr = side[4].transform.parent.GetComponent<PivotRotation>();
        pr.startAutoRotate(side, angle);
    }
}

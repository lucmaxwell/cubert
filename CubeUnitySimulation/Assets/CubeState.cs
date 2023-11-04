using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeState : MonoBehaviour
{

    public List<GameObject> top = new List<GameObject>();
    public List<GameObject> backRight = new List<GameObject>();
    public List<GameObject> backLeft = new List<GameObject>();
    public List<GameObject> frontRight = new List<GameObject>();
    public List<GameObject> frontLeft = new List<GameObject>();
    public List<GameObject> bottom = new List<GameObject>();

    public static bool isAutoRotating = false;

    public static bool started = false;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        // printState();
    }

    void printState()
    {
        print("top: " + string.Join(", ", top));
        print("backRight: " + string.Join(", ", backRight));
        print("backLeft: " + string.Join(", ", backLeft));
        print("frontRight: " + string.Join(", ", frontRight));
        print("frontLeft: " + string.Join(", ", frontLeft));
        print("bottom: " + string.Join(", ", bottom));
    }

    public void PickUp(List<GameObject> cubeSide)
    {
        foreach (GameObject face in cubeSide)
        {
            if (face != cubeSide[4])
            {
                face.transform.parent.transform.parent = cubeSide[4].transform.parent;
            }
        }
        // // print(string.Join(", ", cubeSide));
        // cubeSide[4].transform.parent.GetComponent<PivotRotation>().Rotate(cubeSide);
    }
}

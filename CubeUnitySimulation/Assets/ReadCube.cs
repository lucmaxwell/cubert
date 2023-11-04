using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReadCube : MonoBehaviour
{
    public Transform tTop;
    public Transform tBottom;
    public Transform tBackRight;
    public Transform tBackLeft;
    public Transform tFrontRight;
    public Transform tFrontLeft;


    private List<List<GameObject>> rays;

    private int layerMask = 1 << 6;


    CubeState cubeState;

    // Start is called before the first frame update
    void Start()
    {
        cubeState = FindObjectOfType<CubeState>();

        cubeState.frontLeft = getFaceState(tFrontLeft, 0, 90, 0, Vector3.right);
        cubeState.frontRight = getFaceState(tFrontRight, 0, 0, 0, Vector3.forward);
        cubeState.backLeft = getFaceState(tBackLeft, 0, -180, 0, -Vector3.forward);
        cubeState.backRight = getFaceState(tBackRight, 0, -90, 0, -Vector3.right);
        cubeState.top = getFaceState(tTop, 90, 0, 0, -Vector3.up);
        cubeState.bottom = getFaceState(tBottom, -90, 0, 0, Vector3.up);

        CubeState.started = true;
    }

    // Update is called once per frame
    void Update()
    {
        cubeState.frontLeft = getFaceState(tFrontLeft, 0, 90, 0, Vector3.right);
        cubeState.frontRight = getFaceState(tFrontRight, 0, 0, 0, Vector3.forward);
        cubeState.backLeft = getFaceState(tBackLeft, 0, -180, 0, -Vector3.forward);
        cubeState.backRight = getFaceState(tBackRight, 0, -90, 0, -Vector3.right);
        cubeState.top = getFaceState(tTop, 90, 0, 0, -Vector3.up);
        cubeState.bottom = getFaceState(tBottom, -90, 0, 0, Vector3.up);
    }

    List<GameObject> getFaceState(Transform rayTransform, int x, int y, int z, Vector3 direction)
    {
        List<GameObject> facesHit = new List<GameObject>();

        // Create all points for the face
        for (int row = -1; row <= 1; row++)
        {
            for (int col = -1; col <= 1; col++)
            {
                Vector3 ray = rayTransform.localPosition;
                ray = new Vector3(0 + row, 0 + col, -2);
                ray = Quaternion.Euler(x, y, z) * ray;

                RaycastHit hit;

                if (Physics.Raycast(ray, direction, out hit, Mathf.Infinity, layerMask))
                {
                    Debug.DrawRay(ray, direction * hit.distance, Color.yellow);
                    facesHit.Add(hit.collider.gameObject);
                }
                else
                {
                    Debug.DrawRay(ray, direction * 1000, Color.green);
                }
            }
        }
        
        return facesHit;

        // for (int i = 0; i < 2; i++)
        // {
        //     List<GameObject> facesHit = new List<GameObject>();
        //     Vector3 ray = rayTransform.transform.position;
        //     ray = new Vector3(ray.x + i, ray.y, ray.z);
        //     RaycastHit hit;

        //     if (Physics.Raycast(ray, -rayTransform.up, out hit, Mathf.Infinity, layerMask))
        //     {
        //         Debug.DrawRay(ray, -rayTransform.up * hit.distance, Color.yellow);
        //         facesHit.Add(hit.collider.gameObject);
        //         print(hit.collider.gameObject.name);
        //     }
        //     else
        //     {
        //         Debug.DrawRay(ray, -rayTransform.up * 1000, Color.green);
        //     }
        //     cubeState.top = facesHit;
        // }
    }

    // List<GameObject> getFaceRays(Transform rayTransform, Vector3 direction)
    // {
    //     int CUBES_PER_SIDE = 3;
    //     int rayCount = 0;

    //     List<GameObject> rays = new List<GameObject>();


    // }

}

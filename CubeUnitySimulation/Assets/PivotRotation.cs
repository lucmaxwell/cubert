using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

public class PivotRotation : MonoBehaviour
{
    private List<GameObject> activeSide;
    private Vector3 localForward;
    private Vector3 mouseRef;
    private bool dragging = false;
    private Vector3 rotation;
    private bool autoRotating = false;
    private float speed = 300f;
    private Quaternion targetQuaternion;
    private ReadCube readCube;
    private CubeState cubeState;

    private float SENSITIVITY = 0.4f;


    // Start is called before the first frame update
    void Start()
    {
        readCube = FindObjectOfType<ReadCube>();
        cubeState = FindObjectOfType<CubeState>();

    }

    // Update is called once per frame
    void Update()
    {
        if (dragging)
        {
            SpinSide(activeSide);

            if (Input.GetMouseButtonUp(0))
            {
                dragging = false;
                RotateToRightAngle();
            }
        }

        if (autoRotating)
        {
            AutoRotate();
        }

    }

    private void SpinSide(List<GameObject> side)
    {
        rotation = Vector3.zero;

        Vector3 mouseOffset = (Input.mousePosition - mouseRef);

        if (side.All(cubeState.top.Contains))
        {
            rotation.y = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * -1;
        }
        if (side.All(cubeState.frontRight.Contains))
        {
            rotation.z = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * -1;
        }
        if (side.All(cubeState.backRight.Contains))
        {
            rotation.x = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * 1;
        }
        if (side.All(cubeState.bottom.Contains))
        {
            rotation.y = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * 1;
        }
        if (side.All(cubeState.frontLeft.Contains))
        {
            rotation.z = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * 1;
        }
        if (side.All(cubeState.backLeft.Contains))
        {
            rotation.x = (mouseOffset.x + mouseOffset.y) * SENSITIVITY * -1;
        }

        transform.Rotate(rotation, Space.World);

        mouseRef = Input.mousePosition;
    }

    public void Rotate(List<GameObject> side)
    {
        // print("In rotate:" + string.Join(", ", side));
        activeSide = side;
        // print("Active side is: " + string.Join(", ", activeSide));
        mouseRef = Input.mousePosition;
        dragging = true;

        localForward = Vector3.zero - side[4].transform.parent.transform.localPosition;
    }

    public void RotateToRightAngle()
    {
        Vector3 vec = transform.localEulerAngles;
        vec.x = (float)(Math.Round(vec.x / 90) * 90);
        vec.y = (float)(Math.Round(vec.y / 90) * 90);
        vec.z = (float)(Math.Round(vec.z / 90) * 90);

        targetQuaternion.eulerAngles = vec;

        autoRotating = true;
    }

    private void AutoRotate()
    {
        dragging = false;

        var step = speed * Time.deltaTime;
        transform.localRotation = Quaternion.RotateTowards(transform.localRotation, targetQuaternion, step);

        if (Quaternion.Angle(transform.localRotation, targetQuaternion) <= 1)
        {
            transform.localRotation = targetQuaternion;

            CubeState.isAutoRotating = false;

            autoRotating = false;
            dragging = false;
        }
    }

    public void startAutoRotate(List<GameObject> side, float angle) {
        cubeState.PickUp(side);

        Vector3 localForward = Vector3.zero - side[4].transform.parent.transform.localPosition;
        targetQuaternion = Quaternion.AngleAxis(angle, localForward) * transform.localRotation;
        activeSide = side;
        autoRotating = true;
    }
}

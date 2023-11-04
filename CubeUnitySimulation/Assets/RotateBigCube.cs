using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateBigCube : MonoBehaviour
{
    Vector2 firstPressPos;
    Vector2 secondPressPos;
    Vector2 movementDirection;
    Vector3 previousMousePosition;
    Vector3 mouseDelta;

    public GameObject target;

    float speed = 250f;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        swipe();
        drag();
    }

    void drag()
    {
        if (Input.GetMouseButton(1))
        {
            var DRAG_SCALING_FACTOR = 0.1f;
            mouseDelta = (Input.mousePosition - previousMousePosition) * DRAG_SCALING_FACTOR;
            transform.rotation = Quaternion.Euler(mouseDelta.y, -mouseDelta.x, 0) * transform.rotation;
        }
        else
        {
            if (transform.rotation != target.transform.rotation)
            {
                var step = speed * Time.deltaTime;
                transform.rotation = Quaternion.RotateTowards(transform.rotation, target.transform.rotation, step);
            }
        }
        previousMousePosition = Input.mousePosition;
    }

    void swipe()
    {
        if (Input.GetMouseButtonDown(1))
        {
            firstPressPos = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
        }

        if (Input.GetMouseButtonUp(1))
        {
            secondPressPos = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
            movementDirection = new Vector2(secondPressPos.x - firstPressPos.x, secondPressPos.y - firstPressPos.y);
            movementDirection.Normalize();
            if (isLeftMovement(movementDirection))
            {
                target.transform.Rotate(0, 90, 0, Space.World);
            }
            else if (isRightMovement(movementDirection))
            {
                target.transform.Rotate(0, -90, 0, Space.World);
            }
            else if (isUpLeftMovement(movementDirection))
            {
                target.transform.Rotate(0, 0, 90, Space.World);
            }
            else if (isUpRightMovement(movementDirection))
            {
                target.transform.Rotate(90, 0, 0, Space.World);
            }
            else if (isDownLeftMovement(movementDirection))
            {
                target.transform.Rotate(-90, 0, 0, Space.World);
            }
            else if (isDownRightMovement(movementDirection))
            {
                target.transform.Rotate(0, 0, -90, Space.World);
            }
        }
    }

    bool isLeftMovement(Vector2 movement)
    {
        return movement.x < 0 && movement.y > -0.5f && movement.y < 0.5f;
    }

    bool isRightMovement(Vector2 movement)
    {
        return movement.x > 0 && movement.y > -0.5f && movement.y < 0.5f;
    }

    bool isUpRightMovement(Vector2 movement)
    {
        return movement.x > 0 && movement.y > 0.5;
    }

    bool isUpLeftMovement(Vector2 movement)
    {
        return movement.x < 0 && movement.y > 0.5;
    }

    bool isDownLeftMovement(Vector2 movement)
    {
        return movement.x < 0 && movement.y < -0.5;
    }

    bool isDownRightMovement(Vector2 movement)
    {
        return movement.x > 0 && movement.y < -0.5;
    }
}


//******************************************************************************************************
//
//	@file 		face_rotator.cpp
//	@author 	Matthew Mora
//	@created	Jan 11, 2024
//	@brief		Contains functions to rotate a cube face
//
//******************************************************************************************************

#include "face_rotator.h"

// function prototypes

void rotateFaceCCW(Face face);
void rotateFaceCW(Face face);

void orientFace(Face face);

// temp will be added in another file
void spinCCW();     // for spinning entire cube
void spinCW();
void flipCube();
void spinFaceCCW();   // for spinning face
void spinFaceCW();

void rotateFace(Face face, Dir dir)
{
    orientFace(face);

    switch(dir){
        case(CCW):
            spinFaceCCW();
            break;
        
        case(CW):
            spinFaceCW();
            break;

        default:
            break;
    }
}

void orientFace(Face face)
{
    switch(face){
        case(FRONT):
            // flip once
            flipCube();
            break;
        case(BOTTOM):
            // do nothing
            break;
        case(TOP):
            // flip twice
            flipCube();
            flipCube();
            break;
        case(BACK):
            // spin twice
            spinCCW();
            spinCCW();
            flipCube();
            break;
        case(LEFT):
            spinCCW();
            flipCube();
            break;
        case(RIGHT):
            spinCW();
            flipCube();
            break;
        default:
            break;
    }
}
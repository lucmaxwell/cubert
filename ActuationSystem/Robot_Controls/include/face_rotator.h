//******************************************************************************************************
//
//	@file 		face_rotator.h
//	@author 	Matthew Mora
//	@created	Jan 11, 2024
//	@brief		Contains functions to rotate a cube face
//
//******************************************************************************************************

#ifndef _FACE_ROTATOR
#define _FACE_ROTATOR

enum Face{
    TOP,
    BOTTOM,
    FRONT,
    BACK,
    LEFT,
    RIGHT,
};

enum Dir{
    CW,     // clockwise
    CCW,    // counter-clockwise
};

void rotateFace(Face face, Dir dir);

#endif  // _FACE_ROTATOR
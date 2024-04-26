#pragma once

#include "ofMain.h"
#include "vec4.h"
#include "matrix4.h"
#include "triangle.h"
#include <vector>

class ofApp : public ofBaseApp{
    
public:
    void setup() override;
    void update() override;
    void draw() override;
    void exit() override;
    
    void keyPressed(int key) override;
    void keyReleased(int key) override;
    void mouseMoved(int x, int y ) override;
    void mouseDragged(int x, int y, int button) override;
    void mousePressed(int x, int y, int button) override;
    void mouseReleased(int x, int y, int button) override;
    void mouseScrolled(int x, int y, float scrollX, float scrollY) override;
    void mouseEntered(int x, int y) override;
    void mouseExited(int x, int y) override;
    void windowResized(int w, int h) override;
    void dragEvent(ofDragInfo dragInfo) override;
    void gotMessage(ofMessage msg) override;
    // load the mesh
    std::vector<vec> Orig_vertices;
    std::vector<vec> vertices;
    int numV, numF;
    std::vector<Face> faces;
    void print();
    // Viewing Transformations
    int w, h;
    float width = 4.0; //  [-2, 2]
    float height= 3.0; // [-1.5, 1.5]
    vec cam, g, tUp;
    vec uVec, vVec, wVec;
    float n, f, t, b, r, l;
    std::vector<std::vector<vec>> FrameBuffer; // The first three values are for color, the fourth is for depth (the z buffer)
    matrix M; // the Transformation matrix
    // face, vertex normals
    std::vector<vec> face_normals;
    std::vector<vec> vertex_normals;
    std::vector<std::vector<float>> zBuffer;
    void Face_normals();
    void Vertex_normals();
    // Rasterization
    void drawline(int x0, int y0, int x1, int y1, float z0, float z1);
    void RasterizeFace(int x0, int y0, int x1, int y1, int x2, int y2, float z0, float  z1, float z2, int idx1, int idx2, int idx3, int idxF, vec color);
    vec BaryCentricCoo(int x0, int y0, int x1, int y1, int x2, int y2, int xp, int yp);
    bool IsInsideFace(vec barycentric);
    vec InterpolateColor(vec aClr, vec bClr, vec cClr, vec barycentric);
    float InterpolateDepth(float aDepth, float bDepth, float cDepth, vec barycentric);
    std::vector<int> Interpolate(int i0, int d0, int i1, int d1);
    ofPixels colorPixels;
    ofTexture   texColor;
    // shading
    //vec light = vec(125,400,125,1);
    vec light;
    std::vector<vec> lightDirs;
    std::vector<vec> vDirs;
    std::vector<vec> halfDirs;
    bool flat = true;
    bool Phong = false;
    bool Gourand = false;
    vec PhongShading(vec normal, vec viewDir, vec half, vec ambientColor, vec diffuseColor, vec specularColor);
    vec GouraudShading(vec normal1,vec normal2, vec normal3, vec lightDir, vec half, vec ambientColor, vec diffuseColor, vec specularColor,
                        float intensity1, float intensity2, float intensity3);
    vec InterpolateColor2(vec color1, vec color2, float t);
    // arc ball
    float first_x, first_y, current_x, current_y;
    matrix M_model, M_cam, P, M_orth, M_vp;
    bool arcBall;
    double pi = M_PI;
    float neg_inf =  -std::numeric_limits<float>::infinity();
    float inf     =  std::numeric_limits<float>::infinity();
    vec u;
    matrix last_rotation;
    matrix current_rotation;
    
    // zoom
    bool zoom;
    // wireframe
    bool wireframe = false;
    bool reset = false;
    bool model1 =  true;
    
};



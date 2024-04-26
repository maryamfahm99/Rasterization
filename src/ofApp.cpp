#include "ofApp.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>


//--------------------------------------------------------------
void ofApp::setup(){
    
    arcBall= false;
    numV = numF = 0;
    
    //        ############## 1. Load the mesh & calculate the face and vertex normals ##############
//    std::string objFileName = "/Users/memo2/Desktop/CS248-OF-template/OpenFrameworks/apps/myApps/Assignment3/data/KAUST_Beacon.obj";
//    std::string objFileName = "/Users/memo2/Desktop/CS248-OF-template/OpenFrameworks/apps/myApps/Assignment3/data/teapot.obj";
    std::string objFileName = "/Users/memo2/Desktop/CS248-OF-template/OpenFrameworks/apps/myApps/Assignment3/data/KAUST_Beacon.obj";
    
    //std::string objFileName = "/Users/memo2/Desktop/CS248-OF-template/OpenFrameworks/apps/myApps/Assignment3/data/strip_clean copy.obj";
    std::ifstream objFile(objFileName);

    if (!objFile.is_open()) {
        std::cerr << "Failed to open the OBJ file." << std::endl;
    }
    vec center;
    float minX, minY, minZ, maxX, maxY, maxZ;
    minX = inf; minY = inf; minZ = inf;
    maxX = neg_inf; maxY = neg_inf; maxZ = neg_inf;

    std::string line;
    while (std::getline(objFile, line)) {
        std::istringstream lineStream(line);
        std::string lineType;
        lineStream >> lineType;
        // vertices
        if (lineType == "v") {
            vec vertex = vec();
            lineStream >> vertex.v[0] >> vertex.v[1] >> vertex.v[2];
            //vertex.print();
            Orig_vertices.push_back(vertex);
            minX = std::min(minX, vertex.v[0]);
            maxX = std::max(maxX, vertex.v[0]);
            minY = std::min(minY, vertex.v[1]);
            maxY = std::max(maxY, vertex.v[1]);
            minZ = std::min(minZ, vertex.v[2]);
            maxZ = std::max(maxZ, vertex.v[2]);
            center = center + vertex;
            numV += 1;
        }
        // faces
        else if (lineType == "f") {
            Face face;
            for (int i = 0; i < 3; ++i) {
                lineStream >> face.vIndex[i];
                if (lineStream.peek() == '/') {
                    lineStream.ignore(); // Skip '/'
                    while (isdigit(lineStream.peek())) {
                        lineStream.ignore(); // Ignore characters until a non-digit is encountered
                    }
                }
            }
            faces.push_back(face);
            numF += 1;
        }else {
            // Skip lines that don't start with 'v' or 'f'
            continue;
        }
    }
    vertices = Orig_vertices;
    
    //              ############## 2. Set the model & viewing transformations ##############
    
    w  = 800; h = 700;
    center = center/numV;
    // Model Transformation
    matrix translation = M.translate(-center.x(), -center.y(), -center.z());
    center.print();
    translation.print();
    std::cout << maxX << " m " << minX <<endl;
    std::cout << maxY << " m " << minY <<endl;
    std::cout << maxZ << " m " << minZ <<endl;
//    float scaleX = 1.0 / std::abs((maxX - minX));
//    float scaleY = 1.0 / std::abs((maxY - minY));
//    float scaleZ = 1.0 / std::abs((maxZ - minZ));
    float scaleX = std::abs((maxX - minX));
    float scaleY = std::abs((maxY - minY));
    float scaleZ = std::abs((maxZ - minZ));
//    // Find the maximum dimension
    float max_dimension = std::max(std::max(scaleX, scaleY), scaleZ);
    scaleX =  scaleX/max_dimension ;
    scaleY = scaleY/max_dimension ;
    scaleZ = scaleZ/max_dimension ;
    std::cout << scaleX << " " << scaleY << " " << scaleZ << endl;

    matrix scaling = M.scale(1/max_dimension, 1/max_dimension, 1/max_dimension);
    scaling.print();
    //matrix rotation = M.RotateY(pi/2);
    matrix M_objModel = scaling*translation;
    
    std::cout<< "translate vertices to center and scale this to a unit cube" << endl;
    vertices[0].print();
    for (vec& vertex : vertices) {
        //vertex.print();
        vertex = M_objModel*P*vertex;
        //vertex.print();
    }
    vertices[0].print();
    Orig_vertices = vertices;
    
    // define the eye, gaze, and the view  up vector t
    cam = vec(0,0,1,1);
    g = vec(0,0,0,1) - cam;
    tUp = vec(0,1,0,0);
    light = vec(0,5,0,1);
    // compute uvw
    wVec = g.normalize()*(-1);
    uVec = (tUp.cross(wVec)).normalize();
    vVec = wVec.cross(uVec).normalize();
    
    // create the viewing transformation matrix // M = M_vp * M_per * M_cam
    
    // define n, f, l, r, t, b to create M_vp and M_per
    n = 1;  f = -3;
    t = 1; b = -t;
    r = 1; l = -r;
    
//    vec cen = vec((maxX+minX) / 2, (maxY+minY) / 2, (maxZ+minZ) / 2);
//    vec extents = vec((maxX-minX) / 2, (maxY-minY) / 2, (maxZ-minZ) / 2);
//
//    // Calculate frustum parameters
//    l = cen.x() - extents.x();
//    r = center.x() + extents.x();
//    b = center.y() - extents.y();
//    t = center.y() + extents.y();
//    n = center.z() - extents.z();
//    f = center.z() + extents.z();
//    
//    float obj_aspect_ratio = scaleX/scaleY;
//    std::cout << "aspect " << obj_aspect_ratio<<endl;
//    obj_aspect_ratio = scaleZ/scaleY;
//    std::cout << "aspect " << obj_aspect_ratio<<endl;
//    obj_aspect_ratio = scaleX/scaleY;
//    std::cout << "aspect " << obj_aspect_ratio<<endl;
    
    M_cam = M.SetupCamTransformation(cam, uVec, vVec, wVec);
    M_vp  = M.SetupViewportTransformation(w, h);
    P = M.SetupPersProjTransformation(l, r, b, t, f, n);
    M_orth = M.SetupOrthProjTransformation(l, r, b, t, f, n);
//    M = M_orth*M_cam;
//    matrix M2  = M_orth* P * M_cam;
    M = M_vp *M_orth;
    // Transform the vertices

    std::cout<< "transform vertices from object space to camera space " << endl;
    for (vec& vertex : vertices) {
        //vertex.print();
        vertex = M_cam*vertex;
        //vertex.print();
    }
    light = M_cam*light;
    light.print();
    Face_normals();
    Vertex_normals(); // calculate the buffers for shading
    
    std::cout<< "transform vertices from camera space to 2D" << endl;
    M_model.print();
    vec vv = vertices[0];
    vv.print();
    vv = M_orth* P*vv;
    P.print();
    M_orth.print();
    vv.print();
    for (vec& vertex : vertices) {
        //vertex.print();
        vertex = M*P*vertex;
        //vertex.print();
    }
    
    
    //               ########################### Rasterization ###########################
    

    // initialize the Framebuffers
    FrameBuffer = std::vector<std::vector<vec>> (w, std::vector<vec>(h,vec(255, 255, 255, neg_inf)));
    zBuffer = std::vector<std::vector<float>> (w, std::vector<float>(h, inf)); // as color
    
    int idxF = 0;
    for (const Face& face : faces){ // for each triangle, rasterize it and draw the lines
        int idx1, idx2, idx3;
        idx1 = face.vIndex[0] - 1; idx2 = face.vIndex[1] - 1; idx3 = face.vIndex[2] - 1;
        vec v1 = vertices[idx1];
        vec v2 = vertices[idx2];
        vec v3 = vertices[idx3];
        int x0, x1, y0, y1, x2, y2, z0, z1, z2;
        
        x0 = round(v1.x()/v1.w()); x1 = round(v2.x()/v2.w()); y0 = round(v1.y()/v1.w()); y1 = round(v2.y()/v2.w());
        x2 = round(v3.x()/v3.w()); y2 = round(v3.y()/v3.w());
        RasterizeFace(x0, y0, x1, y1, x2, y2, v1.w(), v2.w(), v3.w(), idx1, idx2, idx3,idxF, vec(192,192,192,0));
//        drawline(x0,y0,x1,y1,v1.w(), v2.w());
//        drawline(x1,y1,x2,y2, v2.w(), v3.w());
//        drawline(x2,y2,x0,y0, v3.w(), v1.w());
        idxF += 1;

    }
    
    // ********************* for testing *********************
//    vec v1 = vec(40+200,270+200,1,1.001); vec v2 = vec(200+200,30+200,1,1.001); vec v3 = vec(50+200,50+200,1,1.001);
////    mom.print(); dad.print(); son.print();
////    RasterizeFace(mom.x(),mom.y() , dad.x(), dad.y(), son.x(), son.y());
////    drawline(v1.x(), v1.y(), v2.x(), v2.y());
////    drawline(v3.x(), v3.y(), v2.x(), v2.y());
////    drawline(v3.x(), v3.y(), v1.x(), v1.y());
//    int x0, x1, y0, y1, x2, y2, z0, z1, z2;
//    x0 = round(v1.x()/v1.w()); x1 = round(v2.x()/v2.w()); y0 = round(v1.y()/v1.w()); y1 = round(v2.y()/v2.w());
//    x2 = round(v3.x()/v3.w()); y2 = round(v3.y()/v3.w());
//    RasterizeFace(x0, y0, x1, y1, x2, y2, v1.w(), v2.w(), v3.w(), 1, 1, 1, 1);
//    drawline(x0,y0,x1,y1);
//    drawline(x1,y1,x2,y2);
//    drawline(x2,y2,x1,y1);
//    v1 = vec(40+200+100,270+200,4,1); v2 = vec(200+200+100,30+200,4,1); v3 = vec(50+200+100,50+200,4,1);
////    mom.print(); dad.print(); son.print();
////    RasterizeFace(mom.x(),mom.y() , dad.x(), dad.y(), son.x(), son.y());
////    drawline(v1.x(), v1.y(), v2.x(), v2.y());
////    drawline(v3.x(), v3.y(), v2.x(), v2.y());
////    drawline(v3.x(), v3.y(), v1.x(), v1.y());
//    x0 = round(v1.x()/v1.w()); x1 = round(v2.x()/v2.w()); y0 = round(v1.y()/v1.w()); y1 = round(v2.y()/v2.w());
//    x2 = round(v3.x()/v3.w()); y2 = round(v3.y()/v3.w());
//    RasterizeFace(x0, y0, x1, y1, x2, y2, v1.w(), v2.w(), v3.w(), 1, 1, 1, 1);
//    drawline(x0,y0,x1,y1);
//    drawline(x1,y1,x2,y2);
//    drawline(x2,y2,x1,y1);
    

    
    // show image
    colorPixels.allocate(w, h, OF_PIXELS_RGB);
    for (int y = 0; y < h; ++y) { // Assuming 'h' is the height of the frame buffer
        for (int x = 0; x < w; ++x) { // Assuming 'w' is the width of the frame buffer
            // Access or modify the 'vec' object at position (x, y)
            vec pixel = FrameBuffer[x][y];
//            colorPixels.setColor(x, h - y - 1, ofColor(pixel.x(),pixel.y(),pixel.z()));
            colorPixels.setColor(x, y, ofColor(pixel.x(),pixel.y(),pixel.z()));
        }
    }
    texColor.allocate(colorPixels);
}

void ofApp::drawline(int x0, int y0, int x1, int y1, float z0, float z1){
    float neg_inf =  -std::numeric_limits<float>::infinity();
    if (abs(x1 - x0) > abs(y1 - y0)) {
        // Line is horizontal-ish
        // Make sure x0 < x1
        if (x0 > x1) {
            swap(x0,x1);
            swap(y0,y1);
            swap(z0, z1);
            
        }
        int yyy;
        std::vector<int> ys = Interpolate(x0, y0, x1, y1);
        for (int x = x0; x<x1; ++x) {
            yyy =  ys[x-x0];
            
            //FrameBuffer[x][yyy] = vec(0,0,0,0);
            
            float newDepth = Interpolate(x0, z0, x1, z1)[x - x0];
            if (newDepth < zBuffer[x][yyy]){
                zBuffer[x][yyy] = newDepth;
                FrameBuffer[x][yyy] = vec(0,0,0,0);
            }
//
//            if (newDepth < FrameBuffer[x][yyy].v[3]) {
//                FrameBuffer[x][yyy] = vec(0, 0, 0, newDepth);
//            }
        }
    } else {
        // Line is vertical-ish
        // Make sure y0 < y1
        if (y0 > y1) {
            swap(x0,x1);
            swap(y0,y1);
            swap(z0, z1);
        }
        std::vector<int> xs = Interpolate(y0, x0, y1, x1);
        for (int y = y0; y<y1; ++y)  {
            float newDepth = Interpolate(y0, z0, y1, z1)[y - y0];

//            if (newDepth > FrameBuffer[xs[y - y0]][y].v[3]) {
//                FrameBuffer[xs[y - y0]][y] = vec(0, 0, 0, newDepth);
//            }
            if (newDepth < zBuffer[xs[y - y0]][y]){
                zBuffer[xs[y - y0]][y] = newDepth;
                FrameBuffer[xs[y - y0]][y] = vec(0,0,0,0);
            }
            //FrameBuffer[xs[y-y0]][y] = vec(0,0,0,0);
        }
    }
}
std::vector<int> ofApp::Interpolate(int i0, int d0, int i1, int d1){
    std::vector<int> values;
    if (i0 == i1) {
        values.push_back(d0);
        return values;
    }
    float a = static_cast<float>(d1 - d0) / (i1 - i0);
    float d = static_cast<float>(d0);
    for (int i = i0; i<i1; ++i) {
        values.push_back(d);
        d = d + a;
    }
    return values;
}

void ofApp::RasterizeFace(int x0, int y0, int x1, int y1, int x2, int y2, float z0, float  z1, float z2, int idx1, int idx2, int idx3,int idxF,vec color){
    // we are assuming the FrameBuffer is initialized. The first three values are for color, the fourth is for depth
    
    if((x0 == x1 && y0 == y1) || (x2 == x1 && y2 == y1)  || (x0 == x2 && y0 == y2) ){
        return;
    }
    
    // Calculate the bounding box of the triangle
    int min_x = std::min(std::min(x0, x1), x2);
    int min_y = std::min(std::min(y0, y1), y2);
    int max_x = std::max(std::max(x0, x1), x2);
    int max_y = std::max(std::max(y0, y1), y2);
    //std::cout << min_y << " "  << max_y << std::endl;

    
//     Clip the bounding box to the screen boundaries
    min_x = std::max(min_x, 0);
    min_y = std::max(min_y, 0);
    max_x = std::min(max_x, static_cast<int>(w-1));
    max_y = std::min(max_y, static_cast<int>(h-1));
    
    // Loop through pixels within the bounding box
    
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            vec barycentric = BaryCentricCoo(x0, y0, x1, y1, x2, y2, x, y);
            // Check if the current pixel is inside the triangle using barycentric coordinates
            if (IsInsideFace(barycentric)) {
                // Interpolate attributes (normal, half, light direction,color) using barycentric coordinates
                
                //          *********************************** shading begin
                vec n, l, h, interpolatedColor;
                l = InterpolateColor(lightDirs[idx1], lightDirs[idx2], lightDirs[idx3], barycentric);
                h  = InterpolateColor(halfDirs[idx1], halfDirs[idx2], halfDirs[idx3], barycentric);
                l = l.normalize(); h.normalize();
                if (flat){
                    n = (face_normals[idxF])*(-1); // flat shading
                    interpolatedColor = PhongShading(n, l, h, vec(0.1,0.1,0.1,0)*255, color, vec(1,1,1,0)*255);
                }else if (Phong){
                    n = InterpolateColor(vertex_normals[idx1]*(-1), vertex_normals[idx2]*(-1), vertex_normals[idx3]*(-1), barycentric);
                    interpolatedColor = PhongShading(n, l, h, vec(0.1,0.1,0.1,0)*255, color, vec(1,1,1,0)*255);
                }else if (Gourand){ //   interpolate the color of each normal
                    vec n2, n3;
                    n = vertex_normals[idx1]*(-1); n2 = vertex_normals[idx2]*(-1); n3 = vertex_normals[idx3]*(-1);
                    interpolatedColor = GouraudShading(n, n2, n3, l, h, vec(0.1,0.1,0.1,0)*255, color, vec(1,1,1,0)*255,std::max(0.0f, n*l), std::max(0.0f, n2*l), std::max(0.0f, n3*l));
                }
                
                // Apply depth testing
                float interpolatedDepth = InterpolateDepth(z0, z1, z2, barycentric);
                //std::cout << "debug" << std::endl;
                //std::cout << y << std::endl;
                //std::cout << interpolatedDepth << " interpolated" << endl;
                if (interpolatedDepth <= this->n && interpolatedDepth >= this->f){
                    if (interpolatedDepth >= FrameBuffer[x][y].v[3]+0.001 ) { // FrameBuffer[y][x].v[3]
                        //std::cout << "arcball  raster??" << endl;
                        //interpolatedColor.print();
                        // Update framebuffer
                        FrameBuffer[x][y] = interpolatedColor;
                        FrameBuffer[x][y].v[3] = interpolatedDepth;
                    }
                }
//                if (wireframe){
//                    FrameBuffer[x][y] = vec(255,255,255,0);
//                    FrameBuffer[x][y].v[3] = interpolatedDepth;
//                }
            }
        }
    }
}

void ofApp::print(){
    // Example: Printing the parsed data
    for (const vec& vertex : vertices) {
        std::cout << "Vertex: " << vertex.v[0] << ", " << vertex.v[1] << ", " << vertex.v[2] << std::endl;
    }
    for (const Face& face : faces) {
        std::cout << "Face: ";
        for (int i = 0; i < 3; ++i) {
            std::cout << "v" << face.vIndex[i];
            if (i < 2) {
                std::cout << " ";  // Add a space to separate vertex indices
            }
        }
        std::cout << std::endl;
    }
    for (const vec& face_normal : face_normals) {
        std::cout << "Face normals: " << face_normal.v[0] << ", " << face_normal.v[1] << ", " << face_normal.v[2] << std::endl;
    }
    for (const vec& vertex_normal : vertex_normals) {
        std::cout << "Vertex normals: " << vertex_normal.v[0] << ", " << vertex_normal.v[1] << ", " << vertex_normal.v[2] << std::endl;
    }
}

void ofApp::Face_normals(){
    for (const Face& face : faces){
        vec v1 = vertices[face.vIndex[0] - 1];
        vec v2 = vertices[face.vIndex[1] - 1];
        vec v3 = vertices[face.vIndex[2] - 1];
        vec normal = ((v1-v2).cross(v3-v2)).normalize();
        //std::cout << "face normal" << endl;
        //normal.print();
        face_normals.push_back(normal);
        
    }
}

void ofApp::Vertex_normals() {
    // Initialize the vertex normals:
    std::vector<vec> vn(numV, vec(0, 0, 0, 0));
    std::vector<float> n_faces(numV, 0.0);

    // For each face
    for (int j = 0; j < numF; ++j) {
        const Face& face = faces[j];
        const vec& face_normal = face_normals[j];

        // Accumulate face normal to each vertex
        for (int i = 0; i < 3; ++i) {
            int index = face.vIndex[i] - 1;
            vn[index] = vn[index] + face_normal;
            n_faces[index] += 1.0f;
        }
    }

    // Normalize the accumulated vertex normals
    for (int i = 0; i < numV; ++i) {
        vec& vertex_normal = vn[i];
        vec normal = (vertex_normal).normalize();

        // Calculate light, view, and half directions
        vec l = (light - vertices[i]).normalize();
        vec v = (vertices[i] * (-1)).normalize();
        v.v[3] = 0;
        vec half = (v + l).normalize();

        // Store directions for later use
        lightDirs.push_back(l);
        vDirs.push_back(v);
        halfDirs.push_back(half);

        // Update the vertex normal
        vertex_normal = normal;
        
    }
    

    // Update the member variable with the computed vertex normals
    vertex_normals = vn;
}

vec ofApp::BaryCentricCoo(int x0, int y0, int x1, int y1, int x2, int y2, int xp, int yp){ // fix this 1
    float alpha, beta, gamma;
    alpha = static_cast<float>( (y1-y2)*xp + (x2-x1)*yp + x1*y2 - x2*y1 )/ static_cast<float>( (y1-y2)*x0 + (x2-x1)*y0 + x1*y2 - x2*y1 );
    beta  = static_cast<float>( (y2-y0)*xp + (x0-x2)*yp + x2*y0 - x0*y2 )/ static_cast<float>( (y2-y0)*x1 + (x0-x2)*y1 + x2*y0 - x0*y2 );
    //gamma = static_cast<float>( (y0-y1)*xp + (x1-x0)*yp + x0*y1 - x1*y0 )/ static_cast<float>( (y0-y1)*xp + (x1-x0)*yp + x0*y1 - x1*y0 );
    gamma = 1 - alpha - beta;
    //vec P = a + (b-a)*alpha + (c-a)*beta;
    return vec(alpha,beta,gamma,0);
}

bool ofApp::IsInsideFace(vec barycentric){
    if (barycentric.x() >= 0.0f-0.001 && barycentric.y() >= 0.0f-0.001 && barycentric.z() >= 0.0f-0.001){
        return true;
    }
    return false;
}
vec ofApp::InterpolateColor(vec aClr, vec bClr, vec cClr, vec barycentric){
    vec interpolatedClr = aClr * barycentric.x() + bClr * barycentric.y() + cClr * barycentric.z();
    return interpolatedClr;
}
float ofApp::InterpolateDepth(float aDepth, float bDepth, float cDepth, vec barycentric){
    float interpolatedDpth = aDepth * barycentric.x() + bDepth * barycentric.y() + cDepth * barycentric.z();
    return interpolatedDpth;
}
vec ofApp::PhongShading(vec normal, vec lightDir, vec half, vec ambientColor, vec diffuseColor, vec specularColor) {
    // Ensure all vectors are normalized
    normal = normal.normalize();
    lightDir = lightDir.normalize();
    //viewDir = viewDir.normalize();
    half = half.normalize();

    // Calculate ambient term
    vec ambient = ambientColor;

    // Calculate diffuse term
    float diffuseFactor = std::max(normal*lightDir, 0.0f);
    vec diffuse = diffuseColor * diffuseFactor;

    // Calculate specular term
    float specularFactor = pow(std::max(normal*half, 0.0f), 32.0f);
    vec specular = specularColor * specularFactor;

    // Final color calculation
    vec color = ambient + diffuse + specular;

    return color;
}

vec ofApp::GouraudShading(vec normal1,vec normal2, vec normal3, vec lightDir, vec half, vec ambientColor, vec diffuseColor, vec specularColor, float intensity1, float intensity2, float intensity3){
    // Interpolate normals
    normal1 = normal1.normalize(); normal2 = normal2.normalize();normal3 = normal3.normalize();
    vec interpolatedNormal = normal1 * intensity1 + normal2 * intensity2 + normal3 * intensity3;
    interpolatedNormal = interpolatedNormal.normalize();

    // Calculate diffuse term
    float diffuseFactor = std::max(0.0f, interpolatedNormal*lightDir);

    // Calculate specular term
    float specularFactor = std::pow(std::max(0.0f, interpolatedNormal*half), 32.0);

    // Calculate final color using the interpolated intensities
    vec diffuse = InterpolateColor2(diffuseColor * diffuseFactor, ambientColor, intensity1);
    vec specular = InterpolateColor2(specularColor * specularFactor, ambientColor, intensity1);

    return diffuse + specular;
}

vec ofApp::InterpolateColor2(vec color1, vec color2, float t) {
    return color1 + (color2 - color1) * t;
}


//--------------------------------------------------------------
void ofApp::update(){
    // update the camera settings? if i do in drag function then maybe no need.
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofSetHexColor(0xffffff);
    texColor.draw(0, 0, w, h);
    // here it updates the rasterization by taking the new transformation when dragged or keys pressed.
    
    
    vertices = Orig_vertices; // Orig_vertices is the vertices for original obj positioned in the center
    
    // Transform the object
    matrix rotation = current_rotation*last_rotation;
    //cam.print();
    //std::cout<< "transform vertices from object space to camera space in draw " << endl;
    //rotation.print();
    //current_rotation.print();
    //last_rotation.print();
    //vertices[0].print();
    
    for (vec& vertex : vertices) {
        //vertex.print();
        vertex = M_cam*rotation*vertex;
        //vertex.print();
    }
    //vertices[0].print();

    
    //Orig_vertices  = vertices;
    light = vec(0,5,0,1);
    //rotation.transpose(); // not working
    light = M_cam*light;
    //light.print();
    Face_normals();
    Vertex_normals(); // calculate the buffers for shading
    
    //std::cout<< "transform vertices from camera space to 2D" << endl;
//    vv = M_vp*M_orth*P*vv;
//    vv.print();

    for (vec& vertex : vertices) {
        //vertex.print();
        vertex = M_vp*M_orth*P*vertex;
        //vertex.print();
    }
    
    
    //   ########################### Rasterization ###########################

    

    // reinitialize the Framebuffers
    FrameBuffer = std::vector<std::vector<vec>> (w, std::vector<vec>(h,vec(255, 255, 255, neg_inf)));
    zBuffer = std::vector<std::vector<float>> (w, std::vector<float>(h, inf)); // depth buffer
 
    
    int idxF = 0;
    
    //vertices[0].print();
    for (const Face& face : faces){ // for each triangle, rasterize it and draw the lines
        int idx1, idx2, idx3;
        idx1 = face.vIndex[0] - 1; idx2 = face.vIndex[1] - 1; idx3 = face.vIndex[2] - 1;
        vec v1 = vertices[idx1];
        vec v2 = vertices[idx2];
        vec v3 = vertices[idx3];
        
     
        
        int x0, x1, y0, y1, x2, y2, z0, z1, z2;
        x0 = round(v1.x()/v1.w()); x1 = round(v2.x()/v2.w()); y0 = round(v1.y()/v1.w()); y1 = round(v2.y()/v2.w());
        x2 = round(v3.x()/v3.w()); y2 = round(v3.y()/v3.w());
        if (wireframe){
            drawline(x0,y0,x1,y1,v1.w(), v2.w());
            drawline(x1,y1,x2,y2, v2.w(), v3.w());
            drawline(x2,y2,x0,y0, v3.w(), v1.w());
            RasterizeFace(x0, y0, x1, y1, x2, y2, v1.w(), v2.w(), v3.w(), idx1, idx2, idx3,idxF, vec(255,255,255,0));
            
        }else{
            RasterizeFace(x0, y0, x1, y1, x2, y2, v1.w(), v2.w(), v3.w(), idx1, idx2, idx3,idxF, vec(192,192,192,0));
        }
        
        idxF += 1;
    }
    
    // show image
    colorPixels.allocate(w, h, OF_PIXELS_RGB);
    for (int y = 0; y < h; ++y) { // Assuming 'h' is the height of the frame buffer
        for (int x = 0; x < w; ++x) { // Assuming 'w' is the width of the frame buffer
            // Access or modify the 'vec' object at position (x, y)
            vec pixel = FrameBuffer[x][y];
            colorPixels.setColor(x, y, ofColor(pixel.x(),pixel.y(),pixel.z()));
        }
    }
        
    texColor.allocate(colorPixels);
    
}

//--------------------------------------------------------------
void ofApp::exit(){

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    matrix m;
    switch(key) {
        case OF_KEY_UP:
            zoom = true;
            std::cout << "Zoom" << endl;
            cam = m.translate(0,0,-0.2)*cam;
            g = vec(0,0,0,1) - cam;
            wVec = g.normalize()*(-1);
            uVec = (tUp.cross(wVec)).normalize();
            vVec = wVec.cross(uVec).normalize();
            M_cam = M.SetupCamTransformation(cam, uVec, vVec, wVec);
            //cam =
            break;
        case OF_KEY_DOWN:
            zoom = true;
            std::cout << "Zoom" << endl;
            cam = m.translate(0,0,0.2)*cam;
            g = vec(0,0,0,1) - cam;
            wVec = g.normalize()*(-1);
            uVec = (tUp.cross(wVec)).normalize();
            vVec = wVec.cross(uVec).normalize();
            M_cam = M.SetupCamTransformation(cam, uVec, vVec, wVec);
            //cam =
            break;
        case OF_KEY_LEFT:
            zoom = true;
            std::cout << "Zoom" << endl;
            cam = m.translate(0.2,0,0)*cam;
            //g = vec(0,0,0,1) - cam;
            wVec = g.normalize()*(-1);
            uVec = (tUp.cross(wVec)).normalize();
            vVec = wVec.cross(uVec).normalize();
            M_cam = M.SetupCamTransformation(cam, uVec, vVec, wVec);
            //cam =
            break;
        case OF_KEY_RIGHT:
            zoom = true;
            std::cout << "Zoom" << endl;
            cam = m.translate(-0.2,0,0)*cam;
            //g = vec(0,0,0,1) - cam;
            wVec = g.normalize()*(-1);
            uVec = (tUp.cross(wVec)).normalize();
            vVec = wVec.cross(uVec).normalize();
            M_cam = M.SetupCamTransformation(cam, uVec, vVec, wVec);
            //cam =
            break;
        case 'w':
            wireframe = !wireframe;
            break;
        case 's':
            flat = !flat;
            break;
        case 'g':
            Gourand= !Gourand;
            Phong = false;
            flat = false;
            break;
        case 'p':
            Phong = !Phong;
            Gourand = false;
            flat = false;
            break;
        case 'r':
            reset = !reset;
            matrix m;
            last_rotation = m;
            current_rotation = m;
            break;
//        case 'm':
//            model1 = !model1;
//            break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    //arcBall =  true;
    //std::cout << "arcball true"<< endl;
    current_x = ((2.0f*x)/w)-1.0f; current_y = 1.0f-((2.0f*y)/h);
    //std::cout << current_x << endl;
    
    
    matrix m;
    current_rotation = m.arcBall(first_x, first_y, current_x, current_y);
//    rotation.print();
//    M_model.print();
//    //matrix translation = m.translate(0, 0, -2);
//    M_model = rotation*M_model;
//    M_model.print();
    
    
    //first_x  = current_x; first_y = current_y;
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    //std::cout<< "mouse pressed" << endl;
    //std::cout << x << " " << y <<  endl;
    first_x = ((2.0f*x)/w)-1.0f; first_y = 1.0f-((2.0f*y)/h);
    std::cout << first_x << " " << first_y << endl;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){ // change this too mouseDragged
    last_rotation = current_rotation*last_rotation;
    matrix m;
    current_rotation = m;
    //arcBall = false;
    //M_model = matrix();

}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

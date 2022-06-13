
#![allow(dead_code)]
#![allow(unused_variables)]


//use core::slice::SlicePattern;

use std::borrow::{Borrow, BorrowMut};
use std::cell::{RefCell, Ref, RefMut};
use std::iter::Product;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};

use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use std::collections::HashMap;
use std::collections::BTreeMap;

use std::ops::{Index, IndexMut, Add, Mul, Sub, Div};


#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x + p.x, self.y + p.y, self.z + p.z)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x * p.x, self.y * p.y, self.z * p.z)
    }
}

impl Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x / p.x, self.y / p.y, self.z / p.z)
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    id : u32,
}

impl Face {
    fn new(id: u32) -> Face {
        Face { id : id}
    }

    fn halfedge(&self, mesh : &Mesh) -> Halfedge {
        let id = mesh.face_halfedge[self.id as usize];
        Halfedge { id : id}
    }

    fn is_boundary(&self, mesh : &Mesh) -> bool {
        let begin = self.halfedge(mesh);
        let mut he = begin;
        loop {
            if he.twin().is_boundary(mesh) {
                return true;
            }
            he = he.next(mesh);
            if he == begin {
                break;
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    id : u32,
}

impl Edge {
    fn new(id: u32) -> Edge {
        Edge {id}
    }

    fn halfedge(&self) -> Halfedge {
        Halfedge::new(self.id * 2)
    }

    fn endpoint1(&self, mesh : &Mesh) -> Vertex {
        self.halfedge().from(mesh)
    }

    fn endpoint2(&self, mesh : &Mesh) -> Vertex {
        self.halfedge().to(mesh)
    }

    fn is_boundary(&self, mesh : &Mesh) -> bool {
        let he = self.halfedge();
        he.is_boundary(mesh) || he.twin().is_boundary(mesh)
    }

    fn is_flip_ok(&self, mesh : &Mesh) -> bool {
        if self.is_boundary(mesh) {
            return false;
        }

        let he = self.halfedge();
        let op = he.twin();

        // check if the flipped edge is already present in the mesh
        let a = he.next(mesh).to(mesh);
        let b = op.next(mesh).to(mesh);

        // This doesn't seem possible ?
        if a == b {
            return false;
        }

        for v in a.adjacent_vertices(mesh) {
            if v == b {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    id : u32,
}

impl Vertex {
    fn new(id: u32) -> Vertex {
        Vertex {id}
    }

    fn halfedge(&self, mesh : &Mesh) -> Option<Halfedge> {
        let id = mesh.vertex_halfedge[self.id as usize];
        if id == std::u32::MAX { None } else { Some(Halfedge::new(id)) }
    }

    fn halfedge_checked(&self, mesh : &Mesh) -> Halfedge {
        self.halfedge(mesh).expect("Vertex::halfedges() called on isolated vertex")
    }

    fn outgoing_halfedges<'a>(&self, mesh : &'a Mesh) -> OutgoingHalfedgeIterator<'a> {
        let he = self.halfedge(mesh);
        OutgoingHalfedgeIterator { begin: he, current: he, mesh : mesh, started: false }
    }

    fn adjacent_vertices<'a>(&self, mesh : &'a Mesh) -> VertexAdjacentVertexIterator<'a> {
        let he = self.halfedge(mesh);
        let he_iterator = OutgoingHalfedgeIterator { begin: he, current: he, mesh : mesh, started: false };
        VertexAdjacentVertexIterator {he_iterator}
    }

    fn is_boundary(&self, mesh : &Mesh) -> bool {
        match self.halfedge(mesh) {
            Some(he) => he.is_boundary(mesh),
            None => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Halfedge {
    id : u32,
}

impl Halfedge {
    fn new(id: u32) -> Halfedge {
        Halfedge { id }
    }

    fn edge(&self) -> Edge {
        Edge::new(self.id / 2)
    }

    fn face(&self, mesh : &Mesh) -> Option<Face> {
        let id = mesh.halfedge_face[self.id as usize];
        if id == std::u32::MAX {
            None
        } else {
            Some(Face::new(id))
        }
    }

    fn twin(&self) -> Halfedge {
        Halfedge::new(self.id ^ 1)
    }

    fn next(&self, mesh : &Mesh) -> Halfedge {
        let id = mesh.next_halfedge[self.id as usize];
        Halfedge::new(id)
    }

    fn prev(&self, mesh : &Mesh) -> Halfedge {
        let id = mesh.prev_halfedge[self.id as usize];
        Halfedge::new(id)
    }

    fn from(&self, mesh : &Mesh) -> Vertex {
        let id = mesh.halfedge_vertex[self.id as usize];
        Vertex::new(id)
    }

    fn to(&self, mesh : &Mesh) -> Vertex {
        self.twin().from(mesh)
    }

    fn is_boundary(&self, mesh : &Mesh) -> bool {
        self.face(mesh).is_none()
    }
}

struct OutgoingHalfedgeIterator<'a>{
    begin : Option<Halfedge>,
    current : Option<Halfedge>,
    mesh : &'a Mesh,
    started : bool,
}

impl<'a> Iterator for OutgoingHalfedgeIterator<'a> {
  type Item = Halfedge;

  fn next(&mut self) -> Option<Self::Item> {
    if self.started {
        let next = self.current.as_ref().expect("started, so begin is not None").twin().next(self.mesh);
        if next == *self.begin.as_ref().expect("started, so begin is not None") {
            return None;
        }
        self.current = Some(next);
        return self.current.clone();
    } else {
        if let Some(begin) = self.begin.as_ref() {
          self.started = true;
          let current = begin.clone();
          self.current = Some(current.clone());
          return Some(current);
        }
        return None;
    }
  }
}

struct VertexAdjacentVertexIterator<'a>{
    he_iterator : OutgoingHalfedgeIterator<'a>,
}

impl<'a> Iterator for VertexAdjacentVertexIterator<'a> {
    type Item = Vertex;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mesh = self.he_iterator.mesh;
        self.he_iterator.next().map(|he| he.to(mesh))
    }
}

struct Callback {
    compress : Box<dyn FnMut(&Vec<u32>)>,
    push_default : Box<dyn FnMut()>,
}

impl Callback {
    fn new<F1, F2>(compress : F1, push_default: F2) -> Callback 
    where F1 : FnMut(&Vec<u32>) + 'static,
          F2 : FnMut() + 'static {
        Callback {
            compress: Box::new(compress),
            push_default: Box::new(push_default),
        }
    }
}

type Callbacks = BTreeMap<usize, Callback>;

#[derive(Clone)]
struct VertexProperty<T> {
    data : Rc<RefCell<Vec<T>>>,
    callbacks : Weak<RefCell<Callbacks>>,
    property_id : usize,
}

impl<T> Drop for VertexProperty<T> {
    fn drop(&mut self) {
        if let Some(callbacks) = self.callbacks.upgrade() {
            let mut callbacks = callbacks.as_ref().borrow_mut();
            callbacks.remove(&self.property_id);
        }
    }
}

impl<T : Default + Clone + 'static> VertexProperty<T> {

    fn invalid() -> VertexProperty<T> {
        VertexProperty {
            data : Rc::new(RefCell::new(Vec::new())),
            callbacks : Weak::new(),
            property_id : 0,
        }
    }

    fn new(mesh : &Mesh) -> VertexProperty<T> {
        let n = mesh.num_vertices();
        let v = vec![T::default(); n];
        let data = Rc::new(RefCell::new(v));

        let callbacks = mesh.callbacks.clone();
        let mut map = callbacks.as_ref().borrow_mut();

        let mut id = 0;
        for (key, _) in map.iter() {
            if *key != id {
                break;
            }
            id += 1;
        }

        let data1 = data.clone();
        let data2 = data.clone();
        let push_default = move || {
            let mut data = data1.as_ref().borrow_mut();
            data.push(T::default());
        };

        let compress = move |index_map : &Vec<u32>| {
            let mut data = data2.as_ref().borrow_mut();
            let mut data_new = Vec::with_capacity(index_map.len());
            for &i in index_map {
                data_new.push(data[i as usize].clone());
            }
            *data = data_new;
        };

        let callback = Callback::new(compress, push_default);
        
        map.insert(n, callback);

        VertexProperty { data : data, callbacks : Rc::downgrade(&callbacks), property_id : 0 }
    }

    fn set(&self, v : Vertex, a : T) {
        let id = v.id;
        let mut data: RefMut<Vec<T>> = self.data.as_ref().borrow_mut();
        data[id as usize] = a;
    }

    fn resize(&mut self, n : usize) {
        let mut data: RefMut<Vec<T>> = self.data.as_ref().borrow_mut();
        data.resize(n, T::default());
    }
}

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename)
        .extension()
        .and_then(OsStr::to_str)
}

fn find_halfedge(v : Vertex, w : Vertex, mesh: &Mesh) -> Option<Halfedge> {
    for he in v.outgoing_halfedges(mesh) {
        if he.to(mesh) == w {
            return Some(he);
        }
    }
    None
}

fn find_free_gap(he : Halfedge, mesh: &Mesh) -> Halfedge {
    let mut he = he;
    loop {
        he = he.next(mesh).twin();
        if !he.is_boundary(mesh) {
            return he;
        }
    }
}

fn parse_vertex(items : &mut std::str::SplitWhitespace) -> Result<usize, io::Error> {
    if let Some(item) = items.next()  {
        let mut end = item.len();
        if let Some(n) = item.find('/') {
            end = n;
        }
        let id = item[..end].parse::<usize>();
        if let Ok(id) = id {
            return Ok(id);
        }
    }
    Err(io::Error::new(io::ErrorKind::InvalidData, "Cannot parse face: invalid vertex format"))
}

fn try_add_face(mesh : &mut Mesh, items : &mut std::str::SplitWhitespace) -> Result<(), io::Error> {
    let i1 = parse_vertex(items)? as u32;
    let i2 = parse_vertex(items)? as u32;
    let i3 = parse_vertex(items)? as u32;
    let face = [Vertex { id : i1 - 1},
                            Vertex { id : i2 - 1},
                            Vertex { id : i3 - 1}];
    println!("Adding face {} {} {}", face[0].id, face[1].id, face[2].id);
    mesh.add_face(&face).unwrap();

    Ok(())
}

#[derive(Clone)]
pub struct Mesh {
    next_halfedge : Vec<u32>,
    prev_halfedge : Vec<u32>,
    halfedge_face : Vec<u32>,
    halfedge_vertex : Vec<u32>,

    face_halfedge : Vec<u32>,
    vertex_halfedge : Vec<u32>,

    points : VertexProperty<Vec3>,

    callbacks : Rc<RefCell<Callbacks>>,
}

impl Mesh {

    fn new() -> Mesh {
        let callbacks = Rc::new(RefCell::new(BTreeMap::new()));
        let mut mesh = Mesh { 
                next_halfedge : vec![], 
                prev_halfedge : vec![], 
                halfedge_face : vec![], 
                halfedge_vertex : vec![], 
                face_halfedge : vec![], 
                vertex_halfedge : vec![], 
                points : VertexProperty::invalid(),
                callbacks : callbacks,
        };

        mesh.points = VertexProperty::new(&mesh);

        mesh
    }

    pub fn load(path : &str) -> Result<Mesh, io::Error> {

        if get_extension_from_filename(path) == Some("obj") {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let mut mesh = Mesh::new();
            for line in reader.lines() {
                let line = line?;
                let mut items = line.split_whitespace();
                match items.next() {
                    Some("v") => {
                        let x = items.next().unwrap().parse::<f64>().unwrap();
                        let y = items.next().unwrap().parse::<f64>().unwrap();
                        let z = items.next().unwrap().parse::<f64>().unwrap();
                        mesh.add_vertex(Vec3::new(x, y, z));
                    },
                    Some("f") => {
                       try_add_face(&mut mesh, &mut items)?;
                    },
                    _ => (),
                }
            }

            println!("Loaded mesh with {} vertices and {} faces", mesh.num_vertices(), mesh.num_faces());
            return Ok(mesh);
        }

        Err(io::Error::new(io::ErrorKind::Other, "Unsupported file type"))
    }

    fn compress(&mut self) {
        let mut index_map = Vec::with_capacity(self.num_vertices());
        let mut vertex_halfedge = Vec::with_capacity(self.num_vertices());
        for i in 0..self.vertex_halfedge.len() {
            let v_id = self.vertex_halfedge[i];
            if v_id != std::u32::MAX {
                index_map.push(i as u32);
                vertex_halfedge.push(v_id);
            }
        }

        self.vertex_halfedge = vertex_halfedge;
        let mut callbacks = self.callbacks.as_ref().borrow_mut();
        for (_, callback) in callbacks.iter_mut() {
            let compress = &mut callback.compress;
            compress(&index_map);
        }
    }

    fn add_face(&mut self, vs : &[Vertex]) -> Option<Face> {
        //assert!(vs.iter().reduce(|b, v| v.id == self.id && b));

        let n = vs.len();
        assert!(n > 2);

        let mut edge_data : Vec<(Option<Halfedge>, bool, bool)> = Vec::with_capacity(n);

        // test for errors
        for i in 0..n {
            let v = vs[i];
            if !v.is_boundary(self) {
                // Nonmanifold vertex
                return None;
            }
            
            let w = vs[(i + 1) % n]; 
            let he = find_halfedge(v, w, self);
            if he.is_some() && !he.as_ref().unwrap().is_boundary(self) {
                // Nonmanifold edge
                return None;
            }
            let is_new = he.is_none();
            let needs_adjustment = false;
            edge_data.push((he, is_new, needs_adjustment));
        }

        let mut next_cache = Vec::with_capacity(6*n);

        // re-link patches if necessary
        for i in 0..n {
            let j = (i + 1) % n;
            
            if let (Some(inner_prev), Some(inner_next)) = (&edge_data[i].0, &edge_data[j].0) {
                if inner_prev.next(self) == *inner_next {
                    let outer_prev = inner_next.twin();
                    let outer_next = inner_prev.twin();

                    let boundary_prev = find_free_gap(outer_prev, self);
                    let boundary_next = boundary_prev.next(self);

                    if boundary_prev == *inner_prev {
                        // Patch re-linking failed
                        return None;
                    }

                    assert!(boundary_prev.is_boundary(self));
                    assert!(boundary_next.is_boundary(self));

                    let patch_start = inner_prev.next(self);
                    let patch_end = inner_next.prev(self);

                    // relink patch
                    next_cache.push((boundary_prev, patch_start));
                    next_cache.push((patch_end, boundary_next));
                    next_cache.push((inner_prev.clone(), inner_next.clone()));
                }
            }
        }

        // Create missing edges 
        for i in 0..n {
            let j = (i + 1) % n;
            let is_new = edge_data[i].1;
            if is_new {
                edge_data[i].0 = Some(self.new_edge(vs[i], vs[j]));
            }
        }

        // Create the face
        let f = self.new_face();

        // setup halfedges
        for i in 0..n {
            let j = (i + 1) % n;

            let v = vs[i];

            let (inner_prev, prev_is_new, _) = edge_data[i].clone();
            let (inner_next, next_is_new, _) = edge_data[j].clone();

            let inner_prev = inner_prev.expect("At this point, all new halfedges should be created");
            let inner_next = inner_next.expect("At this point, all new halfedges should be created");

            // set outer links
            if prev_is_new || next_is_new {

                let outer_prev = inner_next.twin();
                let outer_next = inner_prev.twin();

                match (prev_is_new, next_is_new) {
                    (true, false) => {
                        let boundary_prev = inner_next.prev(self);
                        next_cache.push((boundary_prev, outer_next));
                        self.set_vertex_halfedge(v, outer_next);
                    },
                    (false, true) => {
                        let boundary_next = inner_prev.next(self);
                        next_cache.push((outer_prev, boundary_next));
                        self.set_vertex_halfedge(v, boundary_next);
                    },
                    (true, true) => {
                        if self.vertex_halfedge[v.id as usize] == std::u32::MAX {
                            self.set_vertex_halfedge(v, outer_next);
                            next_cache.push((outer_prev, outer_next));
                        }
                        else {
                            let boundary_next = v.halfedge(self).expect("exists by construction");
                            let boundary_prev = boundary_next.prev(self);
                            next_cache.push((boundary_prev, outer_next));
                            next_cache.push((outer_prev, boundary_next));
                        }
                    },
                    _ => unreachable!(),
                }

                // set inner link
                next_cache.push((inner_prev, inner_next));
            }
            else {
                edge_data[j].2 = v.halfedge(self).expect("vertex halfedge was set when generating edges") == inner_next;
            }

            // set face handle
            self.set_halfedge_face(edge_data[i].0.unwrap(), f);
        }

        for (he1, he2) in next_cache {
            self.set_next_and_prev(he1, he2);
        }

        for ((_, _, needs_adjustment), &v) in edge_data.iter().zip(vs) {
            if *needs_adjustment {
                self.adjust_outgoing_halfedge(v);
            }
        }

        Some(f)
    }

    fn add_vertex(&mut self, p : Vec3) -> Vertex {
        let id = self.num_vertices();
        self.vertex_halfedge.push(std::u32::MAX);
        let v  = Vertex {id : id as u32};
        self.points.set(v, p);
        self.resize_properties_after_add_vertex();
        v
    }

    fn resize_properties_after_add_vertex(&self) {
        let mut callbacks = self.callbacks.as_ref().borrow_mut();
        for (_, cb) in callbacks.iter_mut() {
            let push = &mut cb.push_default;
            push();
        }
    }

    fn flip(&mut self, e : Edge) {
        assert!(e.is_flip_ok(self));

        let a0 = e.halfedge();
        let b0 = a0.twin();
      
        let a1 = a0.next(self);
        let a2 = a1.next(self);
      
        let b1 = b0.next(self);
        let b2 = b1.next(self);
      
        let   va0 = a0.to(self);
        let   va1 = a1.to(self);
      
        let   vb0 = b0.to(self);
        let   vb1 = b1.to(self);
      
        let     fa  = a0.face(self).expect("Face exists if flip is ok");
        let     fb  = b0.face(self).expect("Face exists if flip is ok");
      
        self.set_halfedge_vertex(a0, va1);
        self.set_halfedge_vertex(b0, vb1);
      
        self.set_next_and_prev(a0, a2);
        self.set_next_and_prev(a2, b1);
        self.set_next_and_prev(b1, a0);

        self.set_next_and_prev(b0, b2);
        self.set_next_and_prev(b2, a1);
        self.set_next_and_prev(a1, b0);
      
        self.set_halfedge_face(a1, fb);
        self.set_halfedge_face(b1, fa);
      
        self.set_face_halfedge(fa, a0);
        self.set_face_halfedge(fb, b0);
      
        if va0.halfedge(self).expect("Not isolated") == b0 {
          self.set_vertex_halfedge(va0, a1);
        }
        if vb0.halfedge(self).expect("Not isolated") == a0 {
          self.set_vertex_halfedge(vb0, b1);
        }
    }

    fn split(&mut self, e : Edge, p : Vec3) -> Vertex {
        let v = self.add_vertex(p);
        todo!();
        v
    }

    fn num_vertices(&self) -> usize {
        self.vertex_halfedge.len()
    }

    fn num_faces(&self) -> usize {
        self.face_halfedge.len()
    }

    fn num_edges(&self) -> usize {
        self.halfedge_vertex.len() / 2
    }

    fn num_halfedges(&self) -> usize {
        self.halfedge_vertex.len() 
    }

    // Low level routines
    fn new_edge(&mut self, v1 : Vertex, v2 : Vertex) -> Halfedge {
        let n = self.next_halfedge.len() as u32;

        let he = Halfedge { id : n + 0};
        let op = Halfedge { id : n + 1};

        self.halfedge_face.extend([std::u32::MAX; 2]);
        self.halfedge_vertex.extend([v1.id as u32, v2.id as u32]);
        self.next_halfedge.extend([std::u32::MAX; 2]);
        self.prev_halfedge.extend([std::u32::MAX; 2]);

        he
    }

    fn new_face(&mut self) -> Face {
        let id = self.face_halfedge.len() as u32;
        self.face_halfedge.push(std::u32::MAX);
        Face{id}
    }

    fn set_vertex_halfedge(&mut self, v : Vertex, he : Halfedge) {
        self.vertex_halfedge[v.id as usize] = he.id as u32;
    }

    fn set_face_halfedge(&mut self, f : Face, he : Halfedge) {
        self.face_halfedge[f.id as usize] = he.id as u32;
    }

    fn set_next_and_prev(&mut self, he_prev : Halfedge, he_next : Halfedge) {
        self.next_halfedge[he_prev.id as usize] = he_next.id as u32;
        self.prev_halfedge[he_next.id as usize] = he_prev.id as u32;
    }

    fn set_halfedge_face(&mut self, he : Halfedge, f : Face) {
        self.halfedge_face[he.id as usize] = f.id as u32;
    }

    fn set_halfedge_vertex(&mut self, he : Halfedge, v : Vertex) {
        self.halfedge_vertex[he.id as usize] = v.id as u32;
    }

    fn adjust_outgoing_halfedge(&mut self, v : Vertex) {
        let hes : Vec<Halfedge> = v.outgoing_halfedges(self).collect();
        for he in hes {
            if he.is_boundary(self) {
                self.set_vertex_halfedge(v, he);
            }
        }
    }
}

#[test]
fn load_cube() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets/cube.obj");
    let mesh = Mesh::load(path.to_str().unwrap()).expect("Failed to load cube.obj");
    assert_eq!(mesh.num_vertices(), 8);
    assert_eq!(mesh.num_faces(), 12);
    assert_eq!(mesh.num_edges(), 18);
}

#[test]
fn load_cube_degenerate() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("assets/cube-degenerate.obj");
    let mesh = Mesh::load(path.to_str().unwrap()).expect("Failed to load cube.obj");
    assert_eq!(mesh.num_vertices(), 8);
    assert_eq!(mesh.num_faces(), 12);
    assert_eq!(mesh.num_edges(), 18);
}

#[test]
fn test_boundary() {
    // Add some vertices
    let mut mesh = Mesh::new();

    let vs = [
    mesh.add_vertex(Vec3::new(0., 1., 0.)),
    mesh.add_vertex(Vec3::new(1., 0., 0.)),
    mesh.add_vertex(Vec3::new(2., 1., 0.)),
    mesh.add_vertex(Vec3::new(0.,-1., 0.)),
    mesh.add_vertex(Vec3::new(2.,-1., 0.)),
    mesh.add_vertex(Vec3::new(3., 0., 0.)),

    // Single point
    mesh.add_vertex(Vec3::new(0.,-2., 0.))];

    // Add two faces
    let faces = [
    mesh.add_face(&[vs[0], vs[1], vs[2]]).unwrap(),
    mesh.add_face(&[vs[1], vs[3], vs[4]]).unwrap(),
    mesh.add_face(&[vs[0], vs[3], vs[1]]).unwrap(),
    mesh.add_face(&[vs[2], vs[1], vs[4]]).unwrap(),
    mesh.add_face(&[vs[5], vs[2], vs[4]]).unwrap()];


    /* Test setup:
        0 ==== 2
        |\  0 /|\
        | \  / | \
        |2  1 3|4 5
        | /  \ | /
        |/  1 \|/
        3 ==== 4

        Vertex 6 single
        */


    // Check for boundary vertices
    assert! ( vs[0].is_boundary(&mesh)  ,"Vertex 0 is not boundary!");
    assert!( !vs[1].is_boundary(&mesh)  , "Vertex 1 is boundary!");
    assert! ( vs[2].is_boundary(&mesh)  , "Vertex 2 is not boundary!");
    assert! ( vs[3].is_boundary(&mesh)  , "Vertex 3 is not boundary!");
    assert! ( vs[4].is_boundary(&mesh)  , "Vertex 4 is not boundary!");
    assert! ( vs[5].is_boundary(&mesh)  , "Vertex 5 is not boundary!");
    assert! ( vs[6].is_boundary(&mesh)  , "Singular Vertex 6 is not boundary!");

    // Check the boundary faces
    assert! ( faces[0].is_boundary(&mesh) , "Face 0 is not boundary!");
    assert! ( faces[1].is_boundary(&mesh) , "Face 1 is not boundary!");
    assert! ( faces[2].is_boundary(&mesh) , "Face 2 is not boundary!");
    assert!( !faces[3].is_boundary(&mesh) , "Face 3 is boundary!");
    assert! ( faces[4].is_boundary(&mesh) , "Face 4 is not boundary!");
}

#[test]
fn test_split_and_flip() {

    let mut mesh = Mesh::new();

    let v0 = mesh.add_vertex(Vec3::new(0., 1., 0.));
    let v1 = mesh.add_vertex(Vec3::new(0., 0., 0.));
    let v2 = mesh.add_vertex(Vec3::new(1., 0., 0.));
    let v3 = mesh.add_vertex(Vec3::new(1., 1., 0.));
  
    let f1 = mesh.add_face(&[v0, v1, v3]).unwrap();
    let f2 = mesh.add_face(&[v1, v2, v3]).unwrap();

    assert_eq!(2, mesh.num_faces());
    assert_eq!(4, mesh.num_vertices());
    assert_eq!(5, mesh.num_edges());
    assert_eq!(10, mesh.num_halfedges());

    /* Test setup:

        0 ==== 3
        |     /|
        |    / | 
        |   /  |
        |  /   | 
        | /    |
        1 ==== 2
        
        1. Split edge (1,3) and (0,3):

        0 ==5= 3
        |\  | /|
        | \ |/ | 
        |   4  |
        |  / \ | 
        | /   \|
        1 ==== 2

        2. Flipping (4,5) and (5,3) is not ok.
        3. Flipping (3,4) is ok.
    */
  
    let e = find_halfedge(v1, v3, &mesh).unwrap().edge();

    let v4 = mesh.split(e, Vec3::new(0.5, 0.5, 0.));
    let v5 = v0.adjacent_vertices(&mesh).find(|&v| v != v4 && v != v2).unwrap();

    assert_eq!(4, mesh.num_faces());
    assert_eq!(5, mesh.num_vertices());
    assert_eq!(10, mesh.num_edges());
    assert_eq!(20, mesh.num_halfedges());

    let e45 = find_halfedge(v4, v5, &mesh).unwrap().edge();
    let e53 = find_halfedge(v5, v3, &mesh).unwrap().edge();
    let e34 = find_halfedge(v3, v4, &mesh).unwrap().edge();

    assert!(!e45.is_flip_ok(&mesh));
    assert!(!e53.is_flip_ok(&mesh));
    assert!(e34.is_flip_ok(&mesh));

    mesh.flip(e34);

    let a = e34.halfedge().from(&mesh);
    let b = e34.halfedge().to(&mesh);

    assert_ne!(a,b);
    assert!(a == v5 || a == v2);
    assert!(b == v5 || b == v2);
}
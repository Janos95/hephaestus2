#![allow(dead_code)]
#![allow(unused_variables)]

use core::panic;

use std::cell::RefCell;
use std::str::Chars;
use std::collections::HashMap;
use std::rc::Rc;

use std::path::Path;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use std::ops;

use enum_as_inner::EnumAsInner;

#[derive(Debug, Copy, Clone, PartialEq)]
enum Type {
  Bool,
  String,
  Mesh,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    LeftParen,
    RightParen,
    End,
    Constant(Value),
    If,
    Else,
    Let,
    Fn,
    EqualSign,
    Arrow,
    Comma,
    Minus,
    Colon,
    Print,
    Return,
    String(String),
    Identifier(String),
    TypeIdentifier(Type),
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x + p.x, self.y + p.y, self.z + p.z)
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x * p.x, self.y * p.y, self.z * p.z)
    }
}

impl ops::Div<Vec3> for Vec3 {
    type Output = Vec3;

    fn div(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x / p.x, self.y / p.y, self.z / p.z)
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, p: Vec3) -> Vec3 {
        Vec3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Face {
    id : usize,
    mesh : MeshObject,
}

#[derive(Debug, Clone, PartialEq)]
struct Edge {
    id : usize,
    mesh : MeshObject,
}

#[derive(Debug, Clone, PartialEq)]
struct Vertex {
    id : usize,
    mesh : MeshObject,
}

#[derive(Debug, Clone, PartialEq)]
struct Halfedge {
    id : usize,
    mesh : MeshObject,
}

#[derive(Debug, Clone, PartialEq)]
struct Mesh {

    halfedges : Vec<u32>,
    face_halfedge : Vec<u32>,
    vertex_halfedge : Vec<u32>,

    points : Vec<Vec3>,
}

type MeshObject = Rc<RefCell<Mesh>>;

fn get_extension_from_filename(filename: &str) -> Option<&str> {
    Path::new(filename)
        .extension()
        .and_then(OsStr::to_str)
}

impl Mesh {
    fn new() -> Mesh {
        Mesh {
            halfedges : Vec::new(),
            face_halfedge : Vec::new(),
            vertex_halfedge : Vec::new(),
            points : Vec::new(),
        }
    }

    fn new_shared() -> MeshObject {
        Rc::new(RefCell::new(Mesh::new()))
    }
    
    fn load(path : &str) -> Result<MeshObject, io::Error> {

        if get_extension_from_filename(path) == Some("obj") {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let mesh = Mesh::new_shared();
            for line in reader.lines() {
                let line = line?;
                let mut items = line.split_whitespace();
                match items.next() {
                    Some("v") => {
                        let x = items.next().unwrap().parse::<f64>().unwrap();
                        let y = items.next().unwrap().parse::<f64>().unwrap();
                        let z = items.next().unwrap().parse::<f64>().unwrap();
                        mesh.borrow_mut().add_vertex(mesh.clone(), Vec3::new(x, y, z));
                    },
                    Some("f") => {
                        let i1 = items.next().unwrap().parse::<usize>().unwrap();
                        let i2 = items.next().unwrap().parse::<usize>().unwrap();
                        let i3 = items.next().unwrap().parse::<usize>().unwrap();
                        let vs = [Vertex { id : i1 - 1, mesh : Rc::clone(&mesh) },
                                  Vertex { id : i2 - 1, mesh : Rc::clone(&mesh) },
                                  Vertex { id : i3 - 1, mesh : Rc::clone(&mesh) }];
                        mesh.borrow_mut().add_face(mesh.clone(), &vs);
                    },
                    _ => (),
                }
            }
            println!("Loaded mesh with {} vertices and {} faces", mesh.borrow().num_vertices(), mesh.borrow().num_faces());
            return Ok(mesh);
        }

        Err(io::Error::new(io::ErrorKind::Other, "Unsupported file type"))
    }

    fn add_face(&mut self, mesh : MeshObject, vs : &[Vertex]) -> Face {
        let id = self.face_halfedge.len();
        self.face_halfedge.push(0);
        Face { id : id, mesh : mesh }
    }

    fn add_vertex(&mut self, mesh : MeshObject, p : Vec3) -> Vertex {
        let id = self.points.len();
        self.points.push(p);
        Vertex {
            id,
            mesh : mesh, 
        }
    }

    fn flip(&mut self, e : &Edge) {

    }

    fn num_vertices(&self) -> usize {
        self.points.len()
    }

    fn num_faces(&self) -> usize {
        self.face_halfedge.len()
    }
}

//struct MeshObject {
//    mesh : Mesh,
//}
//
//impl Object for MeshObject {
//  fn name() -> &'static str {
//    "Mesh"
//  }
//
//  // will need this later for autocomplete
//  fn get_methods(&self) -> Vec<Function> {
//      //let flip_method = Function::new("flip", &[]);
//      vec![]
//  }
//
//  fn call_method(&mut self, name : &str, args : &[Value]) -> Option<Value> {
//      match name {
//        "flip" => {
//            let e = args.first().unwrap().as_edge().unwrap();
//            return Some(self.mesh.flip(e));
//        },
//        _ => panic!("Mesh object does not support the method called")
//      }
//  }
//
//}

#[derive(Debug, Clone, PartialEq, EnumAsInner)]
enum Value {
  Real(f64),
  Int(i64),
  Bool(bool),
  String(String),
  Mesh(MeshObject),
  Edge(Edge),
  //Object(Rc<dyn Object>)
}

struct Tokenizer <'a> {
  it : Chars<'a>,
}

fn contains(a : &mut Chars, keyword: &str) -> bool
{
    for c1 in keyword.chars() {
        if let Some(c2) = a.next() {
           if c1 != c2 {
               return false;
           }
        }
    }
    true
}

impl<'a> Tokenizer<'a> {
    fn new(inp : &'a str) -> Tokenizer<'a> {
        Tokenizer {it : inp.chars()}
    }

    fn parse_type_name(&self, it : &Chars<'a>) -> (Option<Token>, Chars<'a>) {
        for (type_name, type_id) in [
            ("Mesh", Type::Mesh), 
            ("String", Type::String), 
            ("bool", Type::Bool), 
            ] {
            let mut it1 = it.clone();
            if contains(&mut it1, type_name) {
                return (Some(Token::TypeIdentifier(type_id)), it1);
            }
        }
        (None, it.clone())
    }

    fn parse_keyword(&self, it : &Chars<'a>) -> (Option<Token>, Chars<'a>) {
        for (keyword, token) in [
            ("if", Token::If), 
            ("else", Token::Else), 
            ("let", Token::Let), 
            ("fn", Token::Fn), 
            ("return", Token::Return), 
            ("print", Token::Print), 
            ("end", Token::End), 
            ("true", Token::Constant(Value::Bool(true))),
            ("false", Token::Constant(Value::Bool(false)))
            ] {
            let mut it1 = it.clone();
            if contains(&mut it1, keyword) {
                return (Some(token), it1);
            }
        }
        (None, it.clone())
    }

  fn skip_comment(&self, it_old : &Chars<'a>) -> Option<Chars<'a>> {
    let mut it = it_old.clone();
    match (it.next(), it.next()) {
        (Some('/'), Some('/')) => {
            while let Some(c) = it.next() {
                if c == '\n' {
                    break;
                }
            }
            return Some(it);
        },
        (_, _) =>  { return None; },
    }
  }

  fn next_impl(&self) -> (Option<Token>, Chars<'a>) {
    let mut it = self.it.clone();
    let mut prev = self.it.clone();
    while let Some(c) = it.next() {
        if c.is_whitespace() {
            prev = it.clone();
            continue;
        }

        if let Some(it_next) = self.skip_comment(&prev) {
            prev = it_next.clone();
            it = it_next;
            continue;
        }

        if let (Some(token), it) = self.parse_type_name(&prev) {
            return (Some(token), it);
        }

        if let (Some(token), it) = self.parse_keyword(&prev) {
            return (Some(token), it);
        }
         
        match c {
            '(' => return (Some(Token::LeftParen), it),
            ')' => return (Some(Token::RightParen), it),
            '=' => return (Some(Token::EqualSign), it),
            ':' => return (Some(Token::Colon), it),
            '-' => {
                let mut it_next = it.clone();
                match it_next.next() {
                    Some('>') => return (Some(Token::Arrow), it_next),
                    _ => return (Some(Token::Minus), it),
                }
            },
            '"' => {
                let mut s = String::new();
                while let Some(c) = it.next() {
                    if c == '"' {
                        return (Some(Token::String(s)), it);
                    }
                    s.push(c);
                }
                panic!("Unterminated string");
            },
            _ => {
                let mut s = String::new();
                s.push(c);
                while let Some(c) = it.next() {
                    if self.is_identifier_terminating(c) {
                        break;
                    }
                    prev = it.clone();
                    s.push(c);
                }
                return (Some(Token::Identifier(s)), prev);
            }
        }
            
    }
    (None, it)
  }

  fn is_identifier_terminating(&self, c : char) -> bool {
    c.is_whitespace() || c == '(' || c == ')' || c == '=' || c == ':' || c == '-' 
  }

  fn next(&mut self) -> Option<Token> {
    let (token, it) = self.next_impl();
    self.it = it;
    token
  }

  fn peek(&self) -> Option<Token> {
    let (token, _) = self.next_impl();
    token
  }
}

struct Environment {
    global_values : HashMap<String, Value>,
    scoped_values : Vec<HashMap<String, Value>>,

    functions : HashMap<String, Rc<FunctionDefNode>>,
}

impl Environment {
    fn new() -> Environment {
        Environment {
            global_values : HashMap::new(),
            scoped_values : vec![HashMap::new()],
            functions : HashMap::new(),
        }
    }

    fn add_function(&mut self, name : String, func_def : Rc<FunctionDefNode>) -> Option<Rc<FunctionDefNode>> {
        self.functions.insert(name, func_def)
    }

    fn get_function(&self, name : &str) -> Option<&Rc<FunctionDefNode>> {
        self.functions.get(name)
    }

    fn get_value(&self, name : &str) -> Option<&Value> {
        if let Some(map) = self.scoped_values.last() {
            if let Some(v) = map.get(name) {
                return Some(v)
            }
        }
        self.global_values.get(name)
    }

    fn set_value(&mut self, name : String, value : Value) -> Option<Value> {
        let map = self.scoped_values.last_mut().unwrap_or(&mut self.global_values);
        map.insert(name, value)
    }

    fn push_scope(&mut self) {
        self.scoped_values.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scoped_values.pop().expect("No scope to pop");
    }
}

trait AstNode  {
    fn evaluate(&self, env : &mut Environment) -> Option<Value>;

    fn children(&self) -> Vec<Rc<dyn AstNode>>;
}

type Node = Rc<dyn AstNode>;

struct PrintNode {
    expr : Node,
}

struct BlockNode {
    statements : Vec<Node>,
}

struct StringLiteralNode {
    value : String,
}

struct ConstantNode {
    value : Value
}

struct IdentifierNode {
    name : String
}

struct IfNode {
    condition : Node,
    consequent : Node,
    alternate : Option<Node>,
}

struct LetNode {
    pattern : String,
    expression : Node,
}

struct FunctionDefNode {
    name : String,
    arguments : Vec<(String, Type)>,
    body : Node,
}

struct FunctionCallNode {
    func_name : String,
    arguments : Vec<Node>,
}

struct BuiltinNode {
    function : Box<dyn Fn(&Environment) -> Option<Value>>,
}

struct ReturnNode {
    return_value : Option<Node>,
}

impl AstNode for BuiltinNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        (self.function)(env)
    }

    fn children(&self) -> Vec<Rc<dyn AstNode>> {
        vec![]
    }
}

impl AstNode for FunctionCallNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let msg = format!("Function {} not found", self.func_name);
        let func_def = env.get_function(&self.func_name).expect(msg.as_str()).clone();
        let arg_decl = &func_def.arguments;

        let mut args = Vec::with_capacity(arg_decl.len());

        // Evaluate the argument expressions in the current scope
        for (arg_node, (arg_name, _)) in self.arguments.iter().zip(arg_decl.iter()) {
            let value = arg_node.evaluate(env).expect("Argument expression must evaluate to a value");
            args.push((arg_name.clone(), value));
        }

        // Push arguments into to new scope and call the functions body
        env.push_scope();
        for (name, value) in args {
            let old = env.set_value(name, value);
            assert!(old.is_none(), "Argument name must be unique");
        }
        let result = func_def.evaluate(env);
        env.pop_scope();
        result
    }

    fn children(&self) -> Vec<Node> {
        self.arguments.clone()
    }
}

impl AstNode for ReturnNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        match self.return_value {
            Some(ref node) => node.evaluate(env),
            None => None,
        }
    }

    fn children(&self) -> Vec<Node> {
        match self.return_value {
            Some(ref node) => vec![node.clone()],
            None => vec![],
        }
    } 
}

impl AstNode for FunctionDefNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        self.body.evaluate(env)
    }

    fn children(&self) -> Vec<Node> {
        vec![self.body.clone()]
    } 
}

impl AstNode for LetNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let v = self.expression.evaluate(env).expect("let expression requires value");
        let o = env.set_value(self.pattern.clone(), v);
        assert!(o.is_none(), "variable redeclaration");
        None
    }

    fn children(&self) -> Vec<Node> {
        vec![self.expression.clone()]
    } 
}

impl AstNode for ConstantNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        Some(self.value.clone())
    }

    fn children(&self) -> Vec<Node> {
        vec![]
    } 
}

impl AstNode for IdentifierNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let msg = format!("Identifier {} not found", self.name);
        Some(env.get_value(&self.name).expect(msg.as_str()).clone())
    }

    fn children(&self) -> Vec<Node> {
        vec![]
    } 
}

impl AstNode for IfNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let cond = self.condition.evaluate(env).unwrap();
        match cond {
            Value::Bool(true) => self.consequent.evaluate(env),
            Value::Bool(false) => {
                if let Some(alt) = &self.alternate {
                    alt.evaluate(env)
                } else {
                    None
                }
            }
            _ => panic!("If condition must be a boolean"),
        }
    }

    fn children(&self) -> Vec<Node> {
        match &self.alternate {
            Some(alt) => vec![self.condition.clone(), self.consequent.clone(), alt.clone()],
            None => vec![self.condition.clone(), self.consequent.clone()],
        }
    }
}

impl AstNode for PrintNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        let v = self.expr.evaluate(env).unwrap();
        match v {
          Value::String(s) => println!("{}", s),
          Value::Real(r) => println!("{}", r),
          Value::Int(i) => println!("{}", i),
          _ => println!("TODO: not printable")
        }
        None
    }

    fn children(&self) -> Vec<Node> {
        vec![self.expr.clone()]
    } 
}

impl AstNode for BlockNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        for s in &self.statements {
            s.evaluate(env);
        }
        None
    }

    fn children(&self) -> Vec<Node> {
        self.statements.clone()
    } 
}

impl AstNode for StringLiteralNode {
    fn evaluate(&self, env : &mut Environment) -> Option<Value> {
        Some(Value::String(self.value.clone()))
    }

    fn children(&self) -> Vec<Node> {
        vec![]
    } 
}

struct Parser {
  tokens : Vec<Token>,
  idx : i64,

  env : Environment,
  entry_point : Option<Node>,
}

fn tokenize(input : &str) -> Vec<Token> {
    let mut tokens = vec![];
    let mut tokenizer = Tokenizer::new(input);
    while let Some(t) = tokenizer.next() {
        tokens.push(t);
    }
    tokens
}

impl Parser {
    fn new(input : &str) -> Parser {
        Parser { tokens : tokenize(input), idx : 0, env : Environment::new(), entry_point: None }
    }

    fn parse_statement(&mut self) -> Option<Node> {
        let t = self.next();
        if let Some(token) = t {
            //println!("statement token {:?}", token);
            match token {
                Token::Print => {
                    self.expect(Token::LeftParen);
                    let expr = self.parse_expression();
                    self.expect(Token::RightParen);
                    return Some(Rc::new(PrintNode { expr }));
                },
                Token::If => {
                    let cond = self.parse_expression();
                    let then_block = self.parse_block();
                    let mut else_block = None;
                    match self.next() {
                        Some(Token::Else) => {
                            else_block = Some(self.parse_block());
                            self.expect(Token::End);
                        },
                        Some(Token::End) => (),
                        _ => panic!("Expected either else or end token, got {:?}", token),
                    }

                    return Some(Rc::new(IfNode {
                        condition : cond,
                        consequent : then_block,
                        alternate : else_block,
                    }));
                },
                Token::Let => {
                    match self.next() {
                        Some(Token::Identifier(name)) => {
                            self.expect(Token::EqualSign);
                            let expr = self.parse_expression();
                            return Some(Rc::new(LetNode{pattern : name, expression : expr}))
                        },
                        _ => panic!("Expected variable name"),
                    }
                },
                Token::Fn => {
                    match self.next() {
                        Some(Token::Identifier(name)) => {
                            self.expect(Token::LeftParen);
                            let mut args = vec![];
                            loop {
                                let (t1,t2,t3) = (self.next(), self.next(), self.next());
                                match (t1, t2, t3) {
                                    (Some(Token::Identifier(arg_name)), Some(Token::Colon), Some(Token::TypeIdentifier(type_id))) => {
                                        args.push((arg_name, type_id));
                                    },
                                    (Some(Token::RightParen), Some(Token::Arrow), Some(Token::TypeIdentifier(_))) => {
                                        break;
                                    },
                                    (_, _, _) => panic!("Expected type identifier, colon, and identifier"),
                                }
                            }
                            let body = self.parse_block();
                            self.expect(Token::End);
                            let is_entry_point = name == "main";
                            let func_node = Rc::new(FunctionDefNode {
                                name : name.clone(),
                                arguments : args,
                                body : body.clone(),
                            });
                            if is_entry_point {
                                self.entry_point = Some(func_node.clone());
                            }
                            self.env.add_function(name, func_node.clone());
                            return Some(func_node as Node);
                        },
                        _ => panic!("Expected variable name"),
                    }
                }
                Token::Return => {
                    let expr = self.parse_expression();
                    return Some(Rc::new(ReturnNode { return_value : Some(expr) }));
                },
                // these tokens end a block
                Token::Else => self.go_back(),
                Token::End => self.go_back(),
                _ => panic!("Unexpected token {:?}", token),
            }
        }
        None
    }

    fn go_back(&mut self) {
        assert!(self.idx > 0);
        self.idx = self.idx - 1;
    }

    fn next(&mut self) -> Option<Token> {
        let idx = self.idx as usize;
        self.idx = self.idx + 1;
        if idx >= self.tokens.len() {
            return None;
        }
        Some(self.tokens[idx].clone()) // Instead of cloning could use a ref
    }

    fn peek(&mut self) -> Option<Token> {
        let idx = self.idx as usize;
        if idx >= self.tokens.len() {
            return None;
        }
        Some(self.tokens[idx].clone()) // Instead of cloning could use a ref
    }

    fn parse_expression(&mut self) -> Node {
        let token = self.next();
        //println!("expression token {:?}", token);
        match token {
            Some(Token::LeftParen) => {
                let expr = self.parse_expression();
                self.expect(Token::RightParen);
                return expr;
            },
            Some(Token::String(v)) => {
                return Rc::new(StringLiteralNode { value : v });
            },
            Some(Token::Constant(v)) => {
                return Rc::new(ConstantNode { value : v });
            },
            Some(Token::Identifier(n)) => {
                match self.peek() {
                    Some(Token::LeftParen) => { // function call
                        self.expect(Token::LeftParen);
                        let mut args = vec![];
                        loop {
                            // TODO: This currently would allow for things like "foo(1,,2 3)". Should make
                            // this more strict at some point.
                            match self.next() {
                                Some(Token::RightParen) => break,
                                Some(Token::Comma) => (),
                                _ => { // try to parse argument expression
                                    self.go_back();
                                    args.push(self.parse_expression());
                                },
                            }
                        }
                        return Rc::new(FunctionCallNode {
                            func_name : n,
                            arguments : args,
                        });
                    },
                    _ => (),
                }
                return Rc::new(IdentifierNode { name : n });
            },
            _ => panic!("Unexpected token {:?}", token),
        }
    }
    
    fn parse_string_literal(&mut self) -> Node {
        let token = self.next().expect("Expected string literal");
        match token {
            Token::String(s) => Rc::new(StringLiteralNode { value : s }),
            _ => panic!("Unexpected token {:?}", token),
        }
    }

    fn parse_block(&mut self) -> Node {
        let mut statements = Vec::new();
        while let Some(stmt) = self.parse_statement() {
            statements.push(stmt);
        }
        Rc::new(BlockNode{statements})
    }

    fn expect(&mut self, token : Token) {
        let t = self.next();
        if let Some(t) = t {
            if t == token {
                return;
            }
        }
        panic!("Expected token {:?}", token)
    }
}

struct Program {
    root : Node,
    entry_point : Node,
    env : Environment,
}

fn load_mesh(env : &Environment) -> Option<Value> {
    let ice_error = "ICE: load_mesh called incorrectly";
    let path = env.get_value("path").expect(ice_error).as_string().expect(ice_error);
    if let Ok(mesh) = Mesh::load(path) {
        return Some(Value::Mesh(mesh));
    }
    None
}

impl Program {
    fn new(script : &str) -> Program {
        let mut parser = Parser::new(script);
        let root = parser.parse_block();
        let mut program = Program{root : root, entry_point : parser.entry_point.unwrap(), env : parser.env};
        program.register_builtins();
        program
    }

    fn run (&mut self) {
        let input = "/Users/janos/hephaestus2/src/bunny.obj".to_string();
        let func_def = self.env.get_function("main");
        let call_main = FunctionCallNode {
            func_name : "main".to_string(),
            arguments : vec![Rc::new(ConstantNode{value : Value::String(input)})],
        };
        call_main.evaluate(&mut self.env);
    }

    fn register_builtins(&mut self) {
        // load_mesh
        {
            let function = Box::new(load_mesh);
            let func_def = Rc::new(FunctionDefNode {
                name : "load_mesh".to_string(),
                arguments : vec![("path".to_string(), Type::String)],
                body : Rc::new(BuiltinNode{function : function}),
            });
            self.env.add_function("load_mesh".to_string(), func_def);
        }
    }
}

fn main() {
  let script = include_str!("import_mesh.hep");

  //let mut tokenizer = Tokenizer::new(script);
  //while let Some(token) = tokenizer.next() {
  //    println!("{:?}", token);
  //}

  let mut program = Program::new(script);
  program.run();

}


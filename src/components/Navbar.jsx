import React from "react";
import { Link, useLocation } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  const location = useLocation();
  return (
    <nav className="navbar">
      <Link to="/" className={location.pathname === "/" ? "active" : ""}>
        Home
      </Link>
      <Link
        to="/about"
        className={location.pathname === "/about" ? "active" : ""}
      >
        About
      </Link>
      <Link
        to="/download"
        className={location.pathname === "/download" ? "active" : ""}
      >
        Download
      </Link>
      <Link
        to="/contact"
        className={location.pathname === "/contact" ? "active" : ""}
      >
        Contact
      </Link>
    </nav>
  );
};

export default Navbar;

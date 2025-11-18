// nav.js
document.addEventListener("DOMContentLoaded", () => {
  const header = `
    <header class="nav">
      <div class="nav-left">
        <div class="logo">ATSPro</div>
      </div>

      <button class="nav-toggle" id="navToggle" aria-label="Toggle navigation">
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
      </button>

      <nav class="nav-links">
        <a href="index.html">Home</a>
        <a href="about.html">How It Works</a>
        <a href="guide.html">ATS Guide</a>
        <a href="samples.html">Sample Reports</a>
        <a href="testimonials.html">Testimonials</a>
        <a href="privacy.html">Privacy</a>
      </nav>
    </header>
  `;

  const footer = `
    <footer class="footer">
      <p>© 2025 ATSPro — Powered by Saurabh ATS Engine v1.0</p>
      <p>Built with FastAPI + AI Resume Analysis</p>
    </footer>
  `;

  // Inject header + footer
  document.body.insertAdjacentHTML("afterbegin", header);
  document.body.insertAdjacentHTML("beforeend", footer);

  // --- ACTIVE NAV HIGHLIGHT (handles /about and /about.html) ---
  let current = window.location.pathname.split("/").pop();
  if (current === "" || !current.includes(".")) {
    current = current + ".html";
    if (current === ".html") current = "index.html";
  }

  document.querySelectorAll(".nav-links a").forEach(a => {
    if (a.getAttribute("href") === current) {
      a.classList.add("nav-active");
    }
  });

  // --- MOBILE TOGGLE ---
  const navToggle = document.getElementById("navToggle");
  const navLinks  = document.querySelector(".nav-links");

  navToggle.addEventListener("click", () => {
    document.body.classList.toggle("nav-open");
  });

  // Close menu when clicking a link (on mobile)
  navLinks.addEventListener("click", (e) => {
    if (e.target.tagName.toLowerCase() === "a") {
      document.body.classList.remove("nav-open");
    }
  });
});

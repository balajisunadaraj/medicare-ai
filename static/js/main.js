/* ===== MediCare - Main JavaScript ===== */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initHeroSlider();
    initScrollAnimations();
    initScrollTopButton();
    initSmoothScrolling();
    initCountUpAnimation();
});

/* ===== NAVBAR ===== */
function initNavbar() {
    const navbar = document.querySelector('.navbar');
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    // Scroll behavior
    const updateNavbar = () => {
        if (window.scrollY > 60) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    };

    window.addEventListener('scroll', updateNavbar);
    updateNavbar();

    // Mobile toggle
    if (navToggle) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('open');
            navLinks.classList.toggle('open');
        });

        // Close menu when clicking a link
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navToggle.classList.remove('open');
                navLinks.classList.remove('open');
            });
        });
    }
}

/* ===== HERO SLIDER ===== */
function initHeroSlider() {
    const slides = document.querySelectorAll('.hero-slide');
    const dots = document.querySelectorAll('.slider-dot');
    const heroTexts = document.querySelectorAll('.hero-slide-text');

    if (slides.length === 0) return;

    let currentSlide = 0;
    let slideInterval;

    const slideData = [
        {
            title: 'AI-Powered <span class="gradient-text">Heart Disease</span> Prediction',
            subtitle: 'Advanced machine learning algorithms analyze your health parameters to predict cardiovascular risks with high accuracy.',
        },
        {
            title: 'Early <span class="gradient-text">Cancer Detection</span> System',
            subtitle: 'Cutting-edge predictive models help identify cancer risk factors early, empowering you with life-saving insights.',
        },
        {
            title: 'Your Health, <span class="gradient-text">Our Priority</span>',
            subtitle: 'Comprehensive medical prediction platform combining AI technology with medical expertise for better health outcomes.',
        }
    ];

    function goToSlide(index) {
        // Remove active from all
        slides.forEach(s => s.classList.remove('active'));
        dots.forEach(d => d.classList.remove('active'));

        currentSlide = index;

        // Activate current
        slides[currentSlide].classList.add('active');
        dots[currentSlide].classList.add('active');

        // Update hero text
        const heroTitle = document.querySelector('.hero-title');
        const heroSubtitle = document.querySelector('.hero-subtitle');

        if (heroTitle && heroSubtitle) {
            // Fade out
            heroTitle.style.opacity = '0';
            heroTitle.style.transform = 'translateY(20px)';
            heroSubtitle.style.opacity = '0';
            heroSubtitle.style.transform = 'translateY(20px)';

            setTimeout(() => {
                heroTitle.innerHTML = slideData[currentSlide].title;
                heroSubtitle.textContent = slideData[currentSlide].subtitle;

                // Fade in
                heroTitle.style.opacity = '1';
                heroTitle.style.transform = 'translateY(0)';
                heroSubtitle.style.opacity = '1';
                heroSubtitle.style.transform = 'translateY(0)';
            }, 400);
        }
    }

    function nextSlide() {
        goToSlide((currentSlide + 1) % slides.length);
    }

    function startAutoSlide() {
        slideInterval = setInterval(nextSlide, 5000);
    }

    function stopAutoSlide() {
        clearInterval(slideInterval);
    }

    // Dot click handlers
    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            stopAutoSlide();
            goToSlide(index);
            startAutoSlide();
        });
    });

    // Pause on hover
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        heroSection.addEventListener('mouseenter', stopAutoSlide);
        heroSection.addEventListener('mouseleave', startAutoSlide);
    }

    // Initialize
    goToSlide(0);
    startAutoSlide();
}

/* ===== SCROLL ANIMATIONS ===== */
function initScrollAnimations() {
    const revealElements = document.querySelectorAll('.reveal, .reveal-left, .reveal-right, .stagger-children');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    });

    revealElements.forEach(el => observer.observe(el));
}

/* ===== SCROLL TO TOP BUTTON ===== */
function initScrollTopButton() {
    const scrollTopBtn = document.querySelector('.scroll-top');
    if (!scrollTopBtn) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            scrollTopBtn.classList.add('visible');
        } else {
            scrollTopBtn.classList.remove('visible');
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

/* ===== SMOOTH SCROLLING ===== */
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navbarHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight;
                window.scrollTo({ top: targetPosition, behavior: 'smooth' });
            }
        });
    });
}

/* ===== COUNT-UP ANIMATION ===== */
function initCountUpAnimation() {
    const statNumbers = document.querySelectorAll('.stat-number');
    if (statNumbers.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.getAttribute('data-count'));
                const suffix = el.getAttribute('data-suffix') || '';
                const prefix = el.getAttribute('data-prefix') || '';
                animateCount(el, target, prefix, suffix);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(el => observer.observe(el));
}

function animateCount(element, target, prefix, suffix) {
    let current = 0;
    const duration = 2000;
    const stepTime = 20;
    const totalSteps = duration / stepTime;
    const increment = target / totalSteps;

    const counter = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(counter);
        }
        element.textContent = prefix + Math.round(current).toLocaleString() + suffix;
    }, stepTime);
}

const reveals = document.querySelectorAll('.reveal');

const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if(entry.isIntersecting){
            entry.target.classList.add('visible');
        }
    });
});

reveals.forEach(el => observer.observe(el));

/* ===== PARALLAX TILT EFFECT ON CARDS ===== */
document.addEventListener('mousemove', (e) => {
    const cards = document.querySelectorAll('.service-card');
    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        } else {
            card.style.transform = '';
        }
    });
});

import 'dart:convert';
import 'dart:math'; // Required for particle math
import 'dart:ui'; // Required for ImageFilter
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> with TickerProviderStateMixin {
  final TextEditingController _textController = TextEditingController();
  
  // Background Animation Controller
  late AnimationController _bgController;
  final List<Particle> _particles = [];
  final Random _random = Random();

  // Logic State variables
  bool _isLoading = false;
  String? _errorMessage;
  bool? _isFake;
  double? _confidence;
  List<Map<String, dynamic>>? _attentionData;
  String? _geminiStatus;
  String? _geminiReason;

  // Narrative text
  final String _contextParagraph = 
      "Global network traffic analysis indicates a 550% surge in synthetic media anomalies. "
      "Algorithmic verification protocols are currently the only defense against "
      "polarized bot-generated discourse. "
      "System integrity depends on real-time data validation.";

  @override
  void initState() {
    super.initState();
    // Initialize Background Animation
    _bgController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 10),
    )..repeat();

    // Generate random particles
    for (int i = 0; i < 35; i++) {
      _particles.add(Particle(_random));
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    _bgController.dispose();
    super.dispose();
  }

  Future<void> _getPrediction() async {
    final String text = _textController.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _isFake = null;
      _confidence = null;
      _attentionData = null;
      _geminiStatus = null;
      _geminiReason = null;
    });

    try {
      String baseUrl = 'http://127.0.0.1:8000';
      if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
        baseUrl = 'http://10.0.2.2:8000';
      }
      final url = Uri.parse('$baseUrl/predict');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'text': text}),
      );

      // Artificial delay for "Scanning" effect
      await Future.delayed(const Duration(milliseconds: 800));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _isFake = data['is_fake'];
          _confidence = data['confidence'];
          _geminiStatus = data['gemini_verification'];
          _geminiReason = data['gemini_reason'];
          if (data['attention_data'] != null) {
            _attentionData = List<Map<String, dynamic>>.from(data['attention_data']);
            _attentionData!.sort((a, b) => (b['score'] as double).compareTo(a['score'] as double));
          }
        });
      } else {
        setState(() => _errorMessage = 'ERR: SERVER_RESP_${response.statusCode}');
      }
    } catch (e) {
      setState(() => _errorMessage = 'ERR: CONNECTION_REFUSED');
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF050510), // Deep Cyber Black
      body: Stack(
        children: [
          // 1. CYBERNETIC BACKGROUND
          Positioned.fill(
            child: AnimatedBuilder(
              animation: _bgController,
              builder: (context, child) {
                return CustomPaint(
                  painter: CyberneticPainter(
                    particles: _particles,
                    animationValue: _bgController.value,
                  ),
                );
              },
            ),
          ),
          
          // Vignette
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                gradient: RadialGradient(
                  center: Alignment.center,
                  radius: 1.2,
                  colors: [Colors.transparent, const Color(0xFF000000).withOpacity(0.9)],
                ),
              ),
            ),
          ),

          // 2. MAIN CONTENT
          SafeArea(
            child: Column(
              children: [
                _buildHeader(),
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.symmetric(horizontal: 32.0, vertical: 24.0),
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        return constraints.maxWidth > 900
                            ? Row(
                                crossAxisAlignment: CrossAxisAlignment.center, 
                                children: [
                                  Expanded(flex: 5, child: _buildInputSection()),
                                  const SizedBox(width: 60),
                                  Expanded(flex: 4, child: _buildContextPanel()),
                                ],
                              )
                            : Column(
                                children: [
                                  _buildInputSection(),
                                  const SizedBox(height: 60),
                                  _buildContextPanel(),
                                ],
                              );
                      },
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: Colors.cyanAccent.withOpacity(0.2))),
        color: Colors.black.withOpacity(0.3),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Flexible(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.hub, color: Colors.cyanAccent, size: 24),
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    'TRUTHLENS',
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.orbitron(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: 2,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              border: Border.all(color: Colors.cyanAccent.withOpacity(0.5)),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              "SYS: ACTIVE",
              style: GoogleFonts.spaceMono(fontSize: 10, color: Colors.cyanAccent),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInputSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          "> ENTER_DATA_STREAM",
          style: GoogleFonts.spaceMono(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Colors.cyanAccent,
          ),
        ),
        const SizedBox(height: 8),
        
        // Cyber Input Box
        Container(
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.4),
            border: Border.all(color: Colors.cyanAccent.withOpacity(0.3)),
            borderRadius: BorderRadius.circular(4), // Sharp corners
            boxShadow: [
              BoxShadow(
                color: Colors.cyanAccent.withOpacity(0.05),
                blurRadius: 10,
                spreadRadius: 1,
              )
            ]
          ),
          child: TextField(
            controller: _textController,
            maxLines: 8,
            style: GoogleFonts.spaceMono(fontSize: 14, color: Colors.white, height: 1.5),
            cursorColor: Colors.cyanAccent,
            decoration: InputDecoration(
              hintText: 'Paste text payload here for analysis...',
              hintStyle: GoogleFonts.spaceMono(color: Colors.white24),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.all(24),
            ),
          ),
        ),
        
        const SizedBox(height: 20),

        // Action Button
        Align(
          alignment: Alignment.centerRight,
          child: SizedBox(
            height: 50,
            width: _isLoading ? 60 : 220,
            child: ElevatedButton(
              onPressed: _isLoading ? null : _getPrediction,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                backgroundColor: Colors.cyanAccent.withOpacity(0.1),
                foregroundColor: Colors.cyanAccent,
                elevation: 0,
                side: const BorderSide(color: Colors.cyanAccent, width: 1.5),
                shape: BeveledRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20, width: 20,
                      child: CircularProgressIndicator(color: Colors.cyanAccent, strokeWidth: 2),
                    )
                  : Text(
                      'RUN_DIAGNOSTIC',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.orbitron(
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.0, 
                        fontSize: 12,
                      ),
                    ),
            ),
          ),
        ),

        // Results Display
        AnimatedSize(
          duration: const Duration(milliseconds: 400),
          curve: Curves.easeOutQuart,
          child: Column(
            children: [
              if (_errorMessage != null)
                Padding(
                  padding: const EdgeInsets.only(top: 24.0),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.redAccent.withOpacity(0.5)),
                      color: Colors.redAccent.withOpacity(0.1),
                    ),
                    child: Text(
                      _errorMessage!,
                      style: GoogleFonts.spaceMono(color: Colors.redAccent),
                    ),
                  ),
                ),
                
              if (_isFake != null)
                Padding(
                  padding: const EdgeInsets.only(top: 32.0),
                  child: _buildResultHUD(),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildResultHUD() {
    final isFake = _isFake!;
    final color = isFake ? Colors.redAccent : Colors.greenAccent;
    final title = isFake ? "THREAT DETECTED" : "INTEGRITY VERIFIED";
    final subTitle = isFake ? "PATTERN MATCH: DISINFORMATION" : "PATTERN MATCH: AUTHENTIC";
    final icon = isFake ? Icons.warning_amber_rounded : Icons.shield_outlined;

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        border: Border.all(color: color.withOpacity(0.6), width: 1),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.1),
            blurRadius: 30,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 40),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.orbitron(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: color,
                        letterSpacing: 1,
                        shadows: [Shadow(color: color, blurRadius: 10)]
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subTitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.spaceMono(
                        fontSize: 12,
                        color: color.withOpacity(0.7),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          
          // Confidence Gauge
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("CONFIDENCE_LEVEL", style: GoogleFonts.spaceMono(color: Colors.white54)),
              Text(
                "${(_confidence! * 100).toStringAsFixed(1)}%",
                style: GoogleFonts.spaceMono(color: color, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: _confidence,
              minHeight: 8,
              backgroundColor: Colors.white10,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          
          if (_attentionData != null && _attentionData!.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text(">> ATTENTION_NODES", style: GoogleFonts.spaceMono(color: Colors.white54, fontSize: 12)),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _attentionData!.take(6).map((item) {
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.white24),
                    color: Colors.white.withOpacity(0.05),
                  ),
                  child: Text(
                    item['token'],
                    style: GoogleFonts.spaceMono(
                      color: Colors.white,
                      fontSize: 12,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
          
          // --- GEMINI SECONDARY VALIDATION ---
          if (_geminiStatus != null && _geminiStatus != "no" && _geminiStatus != "error") ...[
            const SizedBox(height: 24),
            Divider(color: Colors.white24, height: 1),
            const SizedBox(height: 24),
             Container(
               width: double.infinity,
               padding: const EdgeInsets.all(16),
               decoration: BoxDecoration(
                 color: (_geminiStatus == "true" ? Colors.greenAccent : Colors.redAccent).withOpacity(0.05),
                 border: Border.all(color: (_geminiStatus == "true" ? Colors.greenAccent : Colors.redAccent).withOpacity(0.3)),
               ),
               child: Column(
                 crossAxisAlignment: CrossAxisAlignment.start,
                 children: [
                   Row(
                     children: [
                       Icon(Icons.security, size: 16, color: (_geminiStatus == "true" ? Colors.greenAccent : Colors.redAccent)),
                       const SizedBox(width: 8),
                       Text(
                         "SECONDARY VERIFICATION // GEMINI",
                         style: GoogleFonts.spaceMono(
                           color: (_geminiStatus == "true" ? Colors.greenAccent : Colors.redAccent),
                           fontSize: 10,
                           letterSpacing: 2,
                         ),
                       ),
                     ],
                   ),
                   const SizedBox(height: 8),
                    Text(
                     _geminiStatus == "true" ? "VERIFIED: FACTUAL" : "ALERT: CONTRADICTION",
                      style: GoogleFonts.orbitron(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                   if (_geminiReason != null) ...[
                     const SizedBox(height: 8),
                     Text(
                       _geminiReason!,
                       style: GoogleFonts.spaceMono(
                         color: Colors.white70,
                         fontSize: 12,
                         height: 1.4,
                       ),
                     ),
                   ],
                 ],
               ),
             ),
          ],
        ],
      ),
    );
  }

  // --- FIXED CONTEXT PANEL (Stack-based layout to prevent overflow) ---
  Widget _buildContextPanel() {
    return Container(
      alignment: Alignment.centerLeft,
      // Stack allows the text to determine the height, while the line stretches to match.
      child: Stack(
        children: [
          // 1. The Decorative Vertical Line
          Positioned(
            left: 0,
            top: 0,
            bottom: 0,
            width: 3,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.cyanAccent.withOpacity(0),
                    Colors.cyanAccent,
                    Colors.cyanAccent.withOpacity(0),
                  ],
                ),
              ),
            ),
          ),
          
          // 2. The Text Content
          Padding(
            padding: const EdgeInsets.only(left: 24.0), // Spacing from the line
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Row(
                  children: [
                    Container(
                      width: 8, height: 8,
                      decoration: const BoxDecoration(color: Colors.redAccent, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      "LIVE_FEED // GLOBAL_NET",
                      style: GoogleFonts.spaceMono(
                        color: Colors.cyanAccent,
                        fontSize: 12,
                        letterSpacing: 2,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  "GLOBAL THREAT\nVECTOR ANALYSIS",
                  style: GoogleFonts.orbitron(
                    color: Colors.white,
                    fontSize: 32,
                    fontWeight: FontWeight.w900,
                    height: 1.1,
                    letterSpacing: 1.5,
                    shadows: [
                      Shadow(color: Colors.cyanAccent.withOpacity(0.3), blurRadius: 15),
                    ]
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.03),
                    border: Border(left: BorderSide(color: Colors.white.withOpacity(0.1))),
                  ),
                  child: Text(
                    _contextParagraph,
                    style: GoogleFonts.spaceMono(
                      fontSize: 13,
                      height: 1.8,
                      color: Colors.white70,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// --------------------------------------------------------
// ANIMATION & PAINTER CLASSES 
// --------------------------------------------------------

class Particle {
  double x;
  double y;
  double speedX;
  double speedY;
  double size;

  Particle(Random random)
      : x = random.nextDouble(),
        y = random.nextDouble(),
        speedX = (random.nextDouble() - 0.5) * 0.003,
        speedY = (random.nextDouble() - 0.5) * 0.003,
        size = random.nextDouble() * 3 + 1;
        
  void update() {
    x += speedX;
    y += speedY;

    if (x < 0 || x > 1) speedX *= -1;
    if (y < 0 || y > 1) speedY *= -1;
  }
}

class CyberneticPainter extends CustomPainter {
  final List<Particle> particles;
  final double animationValue;

  CyberneticPainter({required this.particles, required this.animationValue});

  @override
  void paint(Canvas canvas, Size size) {
    final width = size.width;
    final height = size.height;

    final paintNode = Paint()
      ..color = Colors.cyanAccent.withOpacity(0.6)
      ..style = PaintingStyle.fill;

    final paintLine = Paint()
      ..color = Colors.cyan.withOpacity(0.2)
      ..strokeWidth = 1.0;

    for (var i = 0; i < particles.length; i++) {
      var p = particles[i];
      p.update(); 

      final dx = p.x * width;
      final dy = p.y * height;

      canvas.drawCircle(Offset(dx, dy), p.size, paintNode);

      for (var j = i + 1; j < particles.length; j++) {
        var p2 = particles[j];
        final dx2 = p2.x * width;
        final dy2 = p2.y * height;

        double dist = sqrt(pow(dx - dx2, 2) + pow(dy - dy2, 2));

        if (dist < 120) {
          paintLine.color = Colors.cyanAccent.withOpacity(1 - (dist / 120));
          canvas.drawLine(Offset(dx, dy), Offset(dx2, dy2), paintLine);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true; 
  }
}
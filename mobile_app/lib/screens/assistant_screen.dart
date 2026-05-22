import 'package:flutter/material.dart';
import '../models/book.dart';
import '../services/ai_service.dart';
import '../services/books_service.dart';
import 'widgets/bottom_nav_bar.dart';
import 'book_detail_screen.dart';
import '../config/translations.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  List<Book>? recommendedBooks;
  ChatMessage({required this.text, required this.isUser, this.recommendedBooks});
}

class AssistantScreen extends StatefulWidget {
  final Book? book;
  const AssistantScreen({Key? key, this.book}) : super(key: key);

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final List<ChatMessage> _messages = [];
  final TextEditingController _controller = TextEditingController();
  bool _isLoading = false;
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          if (widget.book != null) {
            final template = context.t('assistant_greeting_book');
            _messages.add(ChatMessage(
                text: template.replaceAll('{bookTitle}', widget.book!.title),
                isUser: false));
          } else {
            _messages.add(ChatMessage(
                text: context.t('assistant_greeting_general'),
                isUser: false));
          }
        });
      }
    });
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isLoading = true;
    });
    _controller.clear();
    _scrollToBottom();

    final response = await AiService.chat(
      text, 
      widget.book?.uniqueId, 
      bookTitle: widget.book?.title, 
      bookAuthor: widget.book?.author
    );
    
    if (mounted) {
      String reply = response['response'] ?? context.t('assistant_not_understand');
      List<Book>? books;
      
      // Parse [[Title]] blocks
      final regExp = RegExp(r"\[\[(.*?)\]\]");
      final matches = regExp.allMatches(reply);
      if (matches.isNotEmpty) {
        books = [];
        for (final match in matches) {
          final title = match.group(1);
          if (title != null && title.length > 2) {
            final searchResults = await BooksService.search(title, logSearch: false);
            if (searchResults.isNotEmpty) {
              books.add(searchResults.first);
            }
          }
        }
      }

      setState(() {
        _isLoading = false;
        if (response['success'] == true) {
          _messages.add(ChatMessage(
            text: reply.replaceAll("[[", "").replaceAll("]]", ""), 
            isUser: false,
            recommendedBooks: books?.isNotEmpty == true ? books : null,
          ));
        } else {
          _messages.add(ChatMessage(text: context.t('assistant_error'), isUser: false));
        }
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(context.t('kutub_ai'), style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() {
                _messages.clear();
                if (widget.book != null) {
                  final template = context.t('assistant_greeting_book');
                  _messages.add(ChatMessage(
                      text: template.replaceAll('{bookTitle}', widget.book!.title),
                      isUser: false));
                } else {
                  _messages.add(ChatMessage(
                      text: context.t('assistant_greeting_general'),
                      isUser: false));
                }
              });
            },
          )
        ],
      ),
      body: Column(
        children: [
          if (widget.book != null) _buildBookHeader(),
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                return _buildChatBubble(msg);
              },
            ),
          ),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          _buildMessageInput(),
        ],
      ),
      bottomNavigationBar: const AppBottomNavBar(currentIndex: 2),
    );
  }

  Widget _buildBookHeader() {
    if (widget.book == null) return const SizedBox.shrink();
    
    final book = widget.book!;
    final coverUrl = book.coverUrl.isNotEmpty
        ? 'https://wsrv.nl/?url=${Uri.encodeComponent(book.coverUrl)}&w=100&output=webp'
        : '';

    return InkWell(
      onTap: () => Navigator.pop(context),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
          border: Border(
            bottom: BorderSide(color: Theme.of(context).dividerColor.withValues(alpha: 0.1)),
          ),
        ),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: coverUrl.isNotEmpty
                  ? Image.network(
                      coverUrl,
                      width: 50,
                      height: 75,
                      fit: BoxFit.cover,
                    )
                  : Container(
                      width: 50,
                      height: 75,
                      color: Colors.grey[300],
                      child: const Icon(Icons.book),
                    ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    book.title,
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text(
                    book.author,
                    style: TextStyle(color: Theme.of(context).colorScheme.secondary, fontSize: 14),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            Icon(context.isRtl ? Icons.chevron_left : Icons.chevron_right),
          ],
        ),
      ),
    );
  }

  Widget _buildChatBubble(ChatMessage message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: message.isUser 
            ? Theme.of(context).colorScheme.primary 
            : Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(20).copyWith(
            bottomRight: message.isUser ? const Radius.circular(0) : const Radius.circular(20),
            bottomLeft: !message.isUser ? const Radius.circular(0) : const Radius.circular(20),
          ),
        ),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message.text,
              style: TextStyle(
                color: message.isUser 
                  ? Theme.of(context).colorScheme.onPrimary 
                  : Theme.of(context).colorScheme.onSurface,
                fontSize: 16,
              ),
            ),
            if (message.recommendedBooks != null) ...[
              const SizedBox(height: 12),
              const Divider(height: 1),
              const SizedBox(height: 8),
              SizedBox(
                height: 140,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: message.recommendedBooks!.length,
                  itemBuilder: (context, i) {
                    final b = message.recommendedBooks![i];
                    final proxyCover = b.coverUrl.isNotEmpty 
                        ? 'https://wsrv.nl/?url=${Uri.encodeComponent(b.coverUrl)}&w=100&output=webp'
                        : null;
                        
                    return GestureDetector(
                      onTap: () {
                        Navigator.push(context, MaterialPageRoute(builder: (_) => BookDetailScreen(book: b)));
                      },
                      child: Container(
                        width: 90,
                        margin: const EdgeInsets.only(left: 8),
                        child: Column(
                          children: [
                            Expanded(
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: proxyCover != null 
                                    ? Image.network(proxyCover, fit: BoxFit.cover)
                                    : Container(color: Colors.grey, child: const Icon(Icons.book, size: 20)),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              b.title,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildMessageInput() {
    return Container(
      padding: const EdgeInsets.all(8.0),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, -5),
          )
        ],
      ),
      child: SafeArea(
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: _controller,
                decoration: InputDecoration(
                  hintText: context.t('ask_ai_placeholder'),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(30),
                    borderSide: BorderSide.none,
                  ),
                  filled: true,
                  fillColor: Theme.of(context).colorScheme.surfaceContainerHighest,
                  contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                ),
                onSubmitted: (_) => _sendMessage(),
              ),
            ),
            const SizedBox(width: 8),
            CircleAvatar(
              backgroundColor: Theme.of(context).colorScheme.primary,
              child: IconButton(
                icon: const Icon(Icons.send, color: Colors.white),
                onPressed: _sendMessage,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

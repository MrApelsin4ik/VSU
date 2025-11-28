from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.db.models import Q
from django.views.decorators.http import require_GET
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect
from .models import Section, News, NewsImage, NewsAttachment, Announcement
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
import json

import os
from django.conf import settings
from django.db import transaction

from .models import News, Announcement, Course, Section

import datetime

@staff_member_required
def admin_main(request):
    """
    Главная админ-страница с меню.
    """
    return render(request, "admin_panel/main.html")


@staff_member_required
def admin_news_list(request):
    news_list = News.objects.select_related("section").all().order_by("-created_at")
    return render(request, "admin_panel/news_list.html", {
        "news_list": news_list,
    })


@staff_member_required
def admin_announcement_list(request):
    announcement_list = Announcement.objects.select_related("section").all().order_by("-created_at")
    return render(request, "admin_panel/announcement_list.html", {
        "announcement_list": announcement_list,
    })


@staff_member_required
def admin_course_list(request):
    course_list = Course.objects.select_related("section").all().order_by("title")
    return render(request, "admin_panel/course_list.html", {
        "course_list": course_list,
    })


@staff_member_required
def admin_section_list(request):
    section_list = Section.objects.select_related("parent").all().order_by("title")
    return render(request, "admin_panel/section_list.html", {
        "section_list": section_list,
    })


def admin_login(request):
    if request.method == "POST":
        email = (request.POST.get("email") or "").strip().lower()
        password = request.POST.get("password") or ""

        user = authenticate(request, username=email, password=password)

        if user is None:
            messages.error(request, "Неверный email или пароль.", extra_tags="admin_login")
        else:
            if not user.is_staff:
                messages.error(request, "У вас нет прав администратора.", extra_tags="admin_login")
            else:
                login(request, user)
                return redirect(reverse("admin_main"))
        return redirect(reverse("admin_login"))

    return render(request, "admin_panel/admin_login.html")

@login_required
@staff_member_required
def admin_main(request):
    return render(request, "admin_panel/admin_main.html")


@require_GET
def main_page(request):
    page = int(request.GET.get("page", 1))
    ann_page = int(request.GET.get("ann_page", 1))
    per_page = 20
    per_page_ann = 15
    news_qs = News.objects.filter(is_published=True).order_by('-created_at')
    announcements_sidebar = Announcement.objects.filter(is_active=True).order_by('-created_at')
    sections_qs = Section.objects.all().values('id', 'title', 'parent_id', 'section_type', 'relative_url')
    nodes = {}
    for s in sections_qs:
        nodes[s['id']] = {'id': s['id'], 'title': s['title'], 'parent_id': s['parent_id'], 'section_type': s['section_type'], 'relative_url': s['relative_url'], 'children': []}
    for s in sections_qs:
        pid = s['parent_id']
        if pid and pid in nodes:
            nodes[pid]['children'].append(nodes[s['id']])
    sections_tree = [n for n in nodes.values() if not n['parent_id']]

    filter_news = request.GET.getlist('filter_news') or request.GET.getlist('filter_news[]')
    filter_announcement = request.GET.getlist('filter_announcement') or request.GET.getlist('filter_announcement[]')
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')

    if filter_news:
        try:
            ids = [int(i) for i in filter_news if i]
            news_qs = news_qs.filter(section_id__in=ids)
        except:
            pass

    if filter_announcement:
        try:
            ids = [int(i) for i in filter_announcement if i]
            announcements_sidebar = announcements_sidebar.filter(section_id__in=ids)
        except:
            pass

    try:
        if date_from:
            dfrom = datetime.date.fromisoformat(date_from)
            news_qs = news_qs.filter(created_at__date__gte=dfrom)
            announcements_sidebar = announcements_sidebar.filter(created_at__date__gte=dfrom)
        if date_to:
            dto = datetime.date.fromisoformat(date_to)
            news_qs = news_qs.filter(created_at__date__lte=dto)
            announcements_sidebar = announcements_sidebar.filter(created_at__date__lte=dto)
    except:
        pass

    total_news = news_qs.count()
    total_ann = announcements_sidebar.count()
    news_slice = news_qs[(page-1)*per_page:page*per_page]
    ann_slice = announcements_sidebar[(ann_page-1)*per_page_ann:ann_page*per_page_ann]
    has_more_news = total_news > page * per_page
    has_more_ann = total_ann > ann_page * per_page_ann

    context = {
        "sections": Section.objects.all(),
        "sections_tree_json": json.dumps(sections_tree, ensure_ascii=False),
        "news_list": news_slice,
        "announcements_sidebar": ann_slice,
        "has_more_news": has_more_news,
        "has_more_ann": has_more_ann,
    }

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        data = {
            "news": [
                {
                    "id": n.id,
                    "title": n.title,
                    "short_description": n.short_description[:150] + "..." if len(n.short_description) > 150 else n.short_description,
                    "created_at": n.created_at.strftime("%d.%m.%Y %H:%M"),
                    "section_title": n.section.title if n.section else "Без раздела"
                }
                for n in news_slice
            ],
            "announcements": [
                {
                    "id": a.id,
                    "title": a.title,
                    "created_at": a.created_at.strftime("%d.%m.%Y %H:%M")
                }
                for a in ann_slice
            ],
            "has_more_news": has_more_news,
            "has_more_announcements": has_more_ann,
            "page": page,
            "ann_page": ann_page
        }
        return JsonResponse(data, safe=True)

    return render(request, "main.html", context)


@require_GET
def search(request):

    query = request.GET.get("q", "").strip()
    remote = request.GET.get("remote")

    existing_news = request.GET.getlist("existing_news_ids[]") or request.GET.getlist("existing_news_ids") or []
    existing_ann = request.GET.getlist("existing_ann_ids[]") or request.GET.getlist("existing_ann_ids") or []
    existing_courses = request.GET.getlist("existing_course_ids[]") or request.GET.getlist("existing_course_ids") or []


    def to_int_list(lst):
        out = []
        for x in lst:
            try:
                out.append(int(x))
            except (ValueError, TypeError):
                continue
        return out

    existing_news = to_int_list(existing_news)
    existing_ann = to_int_list(existing_ann)
    existing_courses = to_int_list(existing_courses)

    if not query:
        return JsonResponse({"news": [], "announcements": [], "courses": []})

    def build_q_for_model(model, query, possible_fields):
        concrete_fields = {f.name for f in model._meta.get_fields() if getattr(f, "concrete", False)}
        q_obj = None
        for fname in possible_fields:
            if fname in concrete_fields:
                part = Q(**{f"{fname}__icontains": query})
                q_obj = part if q_obj is None else (q_obj | part)
        return q_obj

    # --- News ---
    news_fields = ["title", "short_description", "description", "body", "content"]
    news_q = build_q_for_model(News, query, news_fields)

    print(f"search q=%r, news_q=%r", query, news_q)
    if news_q is not None:
        news_qs = News.objects.filter(is_published=True).filter(news_q).exclude(id__in=existing_news).order_by(
            "-created_at")
    else:
        news_qs = News.objects.none()
    print(f"news count: %d sql: %s", news_qs.count(), str(news_qs.query))

    # --- Announcements ---
    ann_fields = ["title", "short_description", "description", "body", "content"]
    ann_q = build_q_for_model(Announcement, query, ann_fields)
    if ann_q is not None:
        ann_qs = Announcement.objects.filter(is_active=True).filter(ann_q).exclude(id__in=existing_ann).order_by("-created_at")
    else:
        ann_qs = Announcement.objects.none()

    # --- Courses ---
    courses_data = []
    if remote:
        course_fields = ["title", "short_description", "description", "body", "content", "summary"]
        course_q = build_q_for_model(Course, query, course_fields)
        if course_q is not None:
            course_qs = Course.objects.filter(is_active=True).filter(course_q).exclude(id__in=existing_courses).order_by("-created_at")
        else:
            course_qs = Course.objects.none()
    else:
        course_qs = Course.objects.none()


    data = {
        "news": [
            {
                "id": n.id,
                "title": n.title,
                "short_description": (getattr(n, "short_description", "") or getattr(n, "description", "") or "")[:150] + ("..." if (getattr(n, "short_description", "") or getattr(n, "description", "") or "") and len((getattr(n, "short_description", "") or getattr(n, "description", ""))) > 150 else ""),
                "created_at": n.created_at.strftime("%d.%m.%Y %H:%M") if getattr(n, "created_at", None) else ""
            }
            for n in news_qs[:10]
        ],
        "announcements": [
            {
                "id": a.id,
                "title": a.title,
                "body": (getattr(a, "short_description", "") or getattr(a, "body", "") or "")[:150] + ("..." if (getattr(a, "short_description", "") or getattr(a, "body", "")) and len((getattr(a, "short_description", "") or getattr(a, "body", ""))) > 150 else ""),
                "created_at": a.created_at.strftime("%d.%m.%Y %H:%M") if getattr(a, "created_at", None) else ""
            }
            for a in ann_qs[:10]
        ],
        "courses": [
            {
                "id": c.id,
                "title": c.title,
                "short_description": (getattr(c, "short_description", "") or getattr(c, "description", "") or "")[:150] + ("..." if (getattr(c, "short_description", "") or getattr(c, "description", "")) and len((getattr(c, "short_description", "") or getattr(c, "description", ""))) > 150 else ""),
                "created_at": c.created_at.strftime("%d.%m.%Y %H:%M") if getattr(c, "created_at", None) else ""
            }
            for c in course_qs[:10]
        ],
    }

    return JsonResponse(data)


@require_GET
def news_detail(request, pk: int):
    news = get_object_or_404(News, pk=pk, is_published=True)
    return render(request, "news_announcement_detail.html", {"news": news})


@require_GET
def announcement_detail(request, pk: int):
    announcement = get_object_or_404(Announcement, pk=pk, is_active=True)
    return render(request, "news_announcement_detail.html", {"announcement": announcement})









@login_required
@user_passes_test(staff_member_required)
def admin_news_create(request):

    sections = Section.objects.all().order_by("title")
    errors = {}
    old = {}
    success = False

    if request.method == "POST":
        section_id = request.POST.get("section")
        title = (request.POST.get("title") or "").strip()
        short_description = (request.POST.get("short_description") or "").strip()
        body = (request.POST.get("body") or "").strip()
        is_published = request.POST.get("is_published") == "on"


        old = {
            "section": section_id,
            "title": title,
            "short_description": short_description,
            "body": body,
            "is_published": is_published,
        }


        section = None
        if not section_id:
            errors["section"] = "Выберите раздел."
        else:
            try:
                section = Section.objects.get(pk=section_id)
            except Section.DoesNotExist:
                errors["section"] = "Выбранный раздел не найден."

        if not title:
            errors["title"] = "Укажите тему новости."
        if not short_description:
            errors["short_description"] = "Заполните краткое описание."
        if not body:
            errors["body"] = "Заполните текст новости."

        if not errors:

            news = News.objects.create(
                section=section,
                title=title,
                short_description=short_description,
                body=body,
                is_published=is_published,
            )


            images = request.FILES.getlist("images")
            for idx, img in enumerate(images):
                NewsImage.objects.create(
                    news=news,
                    image=img,
                    is_preview=(idx == 0),
                    sort_order=idx,
                )


            attachments = request.FILES.getlist("attachments")
            for f in attachments:
                NewsAttachment.objects.create(
                    news=news,
                    file=f,
                    original_name=f.name,
                )

            success = True
            errors = {}
            old = {}  # очищаем форму

    context = {
        "sections": sections,
        "errors": errors,
        "old": old,
        "success": success,
    }
    return render(request, "admin_panel/news_create.html", context)




try:
    from dotenv import load_dotenv
    load_dotenv('.env')
except Exception as e:
    pass

API_KEY_EXPECTED = os.environ.get('API_KEY')



from rest_framework import serializers, status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Q


class APIKeyHeaderPermission(permissions.BasePermission):
    message = "Invalid or missing API key."

    def has_permission(self, request, view):
        if not API_KEY_EXPECTED:

            return False
        provided = request.headers.get('X-API-KEY') or request.META.get('HTTP_X_API_KEY')
        return bool(provided and provided == API_KEY_EXPECTED)



class CourseImageSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.SerializerMethodField()
    is_preview = serializers.BooleanField()
    sort_order = serializers.IntegerField()

    def get_url(self, obj):
        request = self.context.get('request')
        if not obj.image:
            return None
        return request.build_absolute_uri(obj.image.url)


class FileAttachmentSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.SerializerMethodField()
    original_name = serializers.CharField()

    def get_url(self, obj):
        request = self.context.get('request')
        if not obj.file:
            return None
        return request.build_absolute_uri(obj.file.url)


class CourseFullSerializer(serializers.ModelSerializer):
    images = CourseImageSerializer(many=True)
    attachments = FileAttachmentSerializer(many=True)

    class Meta:
        model = Course
        fields = [
            'id', 'title', 'description', 'description_ai', 'description_ai_l2',
            'created_at', 'updated_at', 'is_active', 'section',
            'images', 'attachments'
        ]


class NewsImageSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.SerializerMethodField()
    is_preview = serializers.BooleanField()
    sort_order = serializers.IntegerField()

    def get_url(self, obj):
        request = self.context.get('request')
        if not obj.image:
            return None
        return request.build_absolute_uri(obj.image.url)


class NewsFullSerializer(serializers.ModelSerializer):
    images = NewsImageSerializer(many=True)
    attachments = FileAttachmentSerializer(many=True)

    class Meta:
        model = News
        fields = [
            'id', 'title', 'short_description', 'body',
            'description_ai', 'description_ai_l2',
            'created_at', 'updated_at', 'is_published', 'section',
            'images', 'attachments'
        ]


class AnnouncementImageSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    url = serializers.SerializerMethodField()
    sort_order = serializers.IntegerField()

    def get_url(self, obj):
        request = self.context.get('request')
        if not obj.image:
            return None
        return request.build_absolute_uri(obj.image.url)


class AnnouncementFullSerializer(serializers.ModelSerializer):
    images = AnnouncementImageSerializer(many=True)
    attachments = FileAttachmentSerializer(many=True)

    class Meta:
        model = Announcement
        fields = [
            'id', 'title', 'body',
            'description_ai', 'description_ai_l2',
            'created_at', 'updated_at', 'is_active', 'section',
            'images', 'attachments'
        ]



class MissingAIListView(APIView):
    permission_classes = [APIKeyHeaderPermission]

    def get(self, request):

        qs_courses = Course.objects.filter(Q(description_ai='') | Q(description_ai_l2=''))
        qs_news = News.objects.filter(Q(description_ai='') | Q(description_ai_l2=''))
        qs_ann = Announcement.objects.filter(Q(description_ai='') | Q(description_ai_l2=''))

        results = []

        for c in qs_courses:
            results.append({
                'model': 'course',
                'id': c.id,
                'title': c.title,
                'created_at': c.created_at,
                'description_ai': c.description_ai,
                'description_ai_l2': c.description_ai_l2,
            })
        for n in qs_news:
            results.append({
                'model': 'news',
                'id': n.id,
                'title': n.title,
                'created_at': n.created_at,
                'description_ai': n.description_ai,
                'description_ai_l2': n.description_ai_l2,
            })
        for a in qs_ann:
            results.append({
                'model': 'announcement',
                'id': a.id,
                'title': a.title,
                'created_at': a.created_at,
                'description_ai': a.description_ai,
                'description_ai_l2': a.description_ai_l2,
            })


        results.sort(key=lambda x: x.get('created_at') or 0)

        return Response({'count': len(results), 'items': results}, status=status.HTTP_200_OK)



class UpdateAIView(APIView):
    permission_classes = [APIKeyHeaderPermission]

    class InputSerializer(serializers.Serializer):
        model = serializers.ChoiceField(choices=['course', 'news', 'announcement'])
        id = serializers.IntegerField()
        description_ai = serializers.CharField(required=False, allow_blank=True)
        description_ai_l2 = serializers.CharField(required=False, allow_blank=True)

    def post(self, request):
        data = request.data
        items = data if isinstance(data, list) else data.get('items') or (data if isinstance(data, dict) and 'model' in data else None)

        if items is None:
            return Response({'detail': 'Неверный формат: ожидается JSON-массив или {"items": [...] } или одиночный объект.'},
                            status=status.HTTP_400_BAD_REQUEST)

        if isinstance(items, dict):
            items = [items]

        serializer = self.InputSerializer(data=items, many=True)
        serializer.is_valid(raise_exception=True)
        validated = serializer.validated_data

        updated = []
        errors = []
        with transaction.atomic():
            for idx, item in enumerate(validated):
                model = item['model']
                pk = item['id']
                fields_to_set = {}
                if 'description_ai' in item:
                    fields_to_set['description_ai'] = item['description_ai']
                if 'description_ai_l2' in item:
                    fields_to_set['description_ai_l2'] = item['description_ai_l2']
                if not fields_to_set:
                    errors.append({'index': idx, 'id': pk, 'model': model, 'error': 'Нет полей для обновления'})
                    continue

                try:
                    if model == 'course':
                        obj = Course.objects.get(pk=pk)
                    elif model == 'news':
                        obj = News.objects.get(pk=pk)
                    elif model == 'announcement':
                        obj = Announcement.objects.get(pk=pk)
                    else:
                        raise Exception('Unknown model')
                except Exception as exc:
                    errors.append({'index': idx, 'id': pk, 'model': model, 'error': f'Not found: {str(exc)}'})
                    continue

                for k, v in fields_to_set.items():
                    setattr(obj, k, v)
                obj.save()
                updated.append({'model': model, 'id': obj.id})

        return Response({'updated': updated, 'errors': errors}, status=status.HTTP_200_OK)



class FullObjectView(APIView):
    permission_classes = [APIKeyHeaderPermission]

    def get(self, request, model: str, pk: int):
        model = model.lower()
        if model == 'course':
            obj = get_object_or_404(Course, pk=pk)
            serializer = CourseFullSerializer(obj, context={'request': request})
        elif model == 'news':
            obj = get_object_or_404(News, pk=pk)
            serializer = NewsFullSerializer(obj, context={'request': request})
        elif model == 'announcement':
            obj = get_object_or_404(Announcement, pk=pk)
            serializer = AnnouncementFullSerializer(obj, context={'request': request})
        else:
            return Response({'detail': 'Unsupported model. Use one of: course, news, announcement.'},
                            status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.data, status=status.HTTP_200_OK)
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from ckeditor.widgets import CKEditorWidget
from django import forms

from .models import (
    CustomUser,
    Section,
    Course,
    CourseImage,
    CourseAttachment,
    News,
    NewsImage,
    NewsAttachment,
    Announcement,
    AnnouncementImage,
    AnnouncementAttachment,
    Feedback,
    FeedbackImage,
    FeedbackAttachment
)




class FeedbackImageInline(admin.TabularInline):
    model = FeedbackImage
    extra = 0
    readonly_fields = ()
    fields = ("image", "sort_order")

class FeedbackAttachmentInline(admin.TabularInline):
    model = FeedbackAttachment
    extra = 0
    fields = ("file", "original_name")

@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("subject", "section", "contact_email", "contact_phone", "created_at", "is_handled")
    list_filter = ("is_handled", "created_at", "section")
    search_fields = ("subject", "description", "contact_email", "contact_phone", "admin_note")
    inlines = (FeedbackImageInline, FeedbackAttachmentInline)
    readonly_fields = ("created_at",)
    actions = ("mark_as_handled",)

    def mark_as_handled(self, request, queryset):
        updated = queryset.update(is_handled=True)
        self.message_user(request, f"{updated} отзыв(ов) отмечено(ы) как обработанные.")
    mark_as_handled.short_description = "Отметить выбранные отзывы как обработанные"

class CustomUserAdmin(BaseUserAdmin):
    model = CustomUser
    list_display = ("email", "username", "full_name", "is_staff", "is_active")
    list_filter = ("is_staff", "is_superuser", "is_active")
    ordering = ("email",)
    search_fields = ("email", "username", "full_name")

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Персональная информация", {"fields": ("username", "full_name")}),
        (
            "Права доступа",
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                )
            },
        ),
    )

    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": (
                    "email",
                    "username",
                    "password1",
                    "password2",
                    "is_staff",
                    "is_superuser",
                    "is_active",
                ),
            },
        ),
    )


    add_form_template = None


admin.site.register(CustomUser, CustomUserAdmin)


@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ("title", "section_type", "parent", 'url', "url_type")
    list_filter = ("section_type","url_type")
    search_fields = ("title",)


class CourseAdminForm(forms.ModelForm):
    description = forms.CharField(widget=CKEditorWidget(), required=False, label="Описание")

    class Meta:
        model = Course
        fields = "__all__"


class CourseImageInline(admin.TabularInline):
    model = CourseImage
    extra = 1
    fields = ("image", "image_preview", "is_preview", "sort_order")
    ordering = ("sort_order",)




class CourseAttachmentInline(admin.TabularInline):
    model = CourseAttachment
    extra = 1
    fields = ("file", "original_name", "download_link")




@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    form = CourseAdminForm
    list_display = ("title", "section", "is_active", "created_at",)
    list_filter = ("is_active", "section")
    search_fields = ("title", "description")
    inlines = (CourseImageInline, CourseAttachmentInline)
    readonly_fields = ("created_at", "updated_at")
    fieldsets = (
        (None, {
            "fields": ("section", "title", "description", "is_active")
        }),
        ("Системное", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",),
        }),
    )



class NewsImageInline(admin.TabularInline):
    model = NewsImage
    extra = 1


class NewsAttachmentInline(admin.TabularInline):
    model = NewsAttachment
    extra = 1

class NewsAdminForm(forms.ModelForm):
    body = forms.CharField(widget=CKEditorWidget(),required=False)

    class Meta:
        model = News
        fields = ['title','section','short_description', 'body', 'is_published','description_ai','description_ai_l2']

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    form = NewsAdminForm
    list_display = ("title", "section", "created_at", "is_published")
    list_filter = ("is_published", "section")
    search_fields = ("title", "short_description", "body")
    inlines = [NewsImageInline, NewsAttachmentInline]

class AnnouncementAdminForm(forms.ModelForm):
    body = forms.CharField(widget=CKEditorWidget(),required=False)

    class Meta:
        model = Announcement
        fields = ['title', 'section', 'body', 'is_active','description_ai','description_ai_l2']


class AnnouncementImageInline(admin.TabularInline):
    model = AnnouncementImage
    extra = 1


class AnnouncementAttachmentInline(admin.TabularInline):
    model = AnnouncementAttachment
    extra = 1


@admin.register(Announcement)
class AnnouncementAdmin(admin.ModelAdmin):
    form = AnnouncementAdminForm
    list_display = ("title", "section", "created_at", "is_active")
    list_filter = ("is_active", "section")
    search_fields = ("title", "body")
    inlines = [AnnouncementImageInline, AnnouncementAttachmentInline]



@admin.register(NewsImage)
class NewsImageAdmin(admin.ModelAdmin):
    list_display = ("news", "is_preview", "sort_order")


@admin.register(NewsAttachment)
class NewsAttachmentAdmin(admin.ModelAdmin):
    list_display = ("news", "original_name")


@admin.register(AnnouncementImage)
class AnnouncementImageAdmin(admin.ModelAdmin):
    list_display = ("announcement", "sort_order")


@admin.register(AnnouncementAttachment)
class AnnouncementAttachmentAdmin(admin.ModelAdmin):
    list_display = ("announcement", "original_name")
